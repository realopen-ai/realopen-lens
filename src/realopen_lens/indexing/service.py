"""Persistent folder indexing operations; no dependency on API or benchmark code."""

from dataclasses import dataclass, asdict, field
from importlib.metadata import version
from collections.abc import Callable
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter

from realopen_lens.storage.catalog import Catalog, STAGES, default_directory, identifier
from realopen_lens.indexing.scanner import EXTENSIONS
from realopen_lens.storage.artifacts import (
    atomic_bytes,
    attempted,
    image_issues,
    load_manifest,
    publish,
    valid_analysis,
)
from realopen_lens.runtime.jobs import environment, recover, stop_worker, supervise, writer_lock
from realopen_lens.models.manager import digest, registry, verify

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexResult:
    job_id: str
    status: str
    collection: str
    files: int
    images: int
    reused_images: int
    failed_files: int
    removed_files: int
    generation: str
    seconds: float
    degraded_files: int = 0
    degraded_images: int = 0
    reused_stages: dict[str, int] = field(default_factory=dict)


def signature(path: Path) -> dict:
    info = path.stat()
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def pipeline_config(options: dict) -> dict:
    from realopen_lens.encoders import onnx as inference
    from realopen_lens.ocr import tesseract as ocr
    from realopen_lens.indexing import processing
    import shutil

    precision = options.get("siglip_precision", "int8")
    if precision not in {"int8", "fp32"}:
        raise ValueError("SigLIP2 precision must be int8 or fp32")
    verify(Path(options["models"]), "int8", precision)
    executable = shutil.which("tesseract")
    if executable is None:
        raise ValueError("Tesseract 5 is required for indexing")
    spec = {
        "version": 1,
        "siglip_precision": precision,
        "model_registry": registry(),
        "inference_sha256": digest(Path(inference.__file__)),
        "ocr_sha256": digest(Path(ocr.__file__)),
        "processing_sha256": digest(Path(processing.__file__)),
        "packages": {
            name: version(name) for name in ["numpy", "Pillow", "onnxruntime", "tokenizers"]
        },
        "ocr_environment": ocr._environment(executable, options["languages"]),
        "languages": options["languages"],
        "threads": options["threads"],
        "thumbnail_side": 512,
        "semantic_characters": 32000,
        "semantic_passages": 16,
        "ocr_characters": 128000,
    }
    import hashlib

    pipeline = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return {**options, "siglip_precision": precision, "pipeline": pipeline, "spec": spec}


def discover(catalog: Catalog, job_id: str, config: dict, previous: list[dict]):
    root, store = Path(config["root"]), catalog.root
    known = {row["path"]: row for row in previous}

    def denied(error):
        raise error  # Never publish a deletion inferred from an unreadable subtree.

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=denied):
        base = Path(directory)
        dirs[:] = sorted(
            name
            for name in dirs
            if name != ".git"
            and not (base / name).is_symlink()
            and not (base / name).resolve().is_relative_to(store)
        )
        for name in sorted(names):
            path = base / name
            if path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            info, image_id, error = None, None, None
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(root) or resolved.is_relative_to(store):
                    raise ValueError("Image symlink escapes the root or points into the index")
                if not resolved.is_file():
                    raise ValueError("Input is not a regular image file")
                info = signature(path)
                if info["size"] > 64 * 1024 * 1024:
                    raise ValueError("Image exceeds 64 MiB encoded size")
                old = known.get(relative)
                if (
                    not config["rehash"]
                    and old
                    and old["image_id"]
                    and json.loads(old["signature"]) == info
                ):
                    image_id = old["image_id"]
                else:
                    image_id = digest(path)
                if signature(path) != info:
                    raise ValueError("File changed while scanning")
            except (OSError, ValueError, RuntimeError) as exc:
                error = str(exc)[:1500]
            catalog.add_file(job_id, relative, image_id, info, error)


def run_index(options: dict, *, stage_runner=supervise) -> IndexResult:
    root, store = Path(options["root"]), Path(options["store"])
    if not root.is_dir() or root == store or root.is_relative_to(store):
        raise ValueError("Select an existing image root outside the index directory")
    started = perf_counter()
    logger.info(
        "Indexing collection %r from %s (SigLIP2 %s; E5 int8)",
        options["collection"],
        root,
        options.get("siglip_precision", "int8"),
    )
    with writer_lock(store):
        catalog = Catalog(store)
        job_id = None
        try:
            recover(catalog)
            collection = catalog.ensure_collection(options["collection"], root)
            logger.info("Verifying local models and Tesseract language packs")
            config = pipeline_config(options)
            previous, active = [], None
            if collection["active_generation"]:
                _, active = catalog.snapshot(options["collection"])
                previous = catalog.files(active["job_id"])
                existing = json.loads(catalog.job(active["job_id"])["config"])["spec"]
                if existing.get("siglip_precision", "int8") != config["spec"]["siglip_precision"]:
                    raise ValueError(
                        "SigLIP2 precision differs from this collection; "
                        "use a separate --index-dir or --collection"
                    )
            job_id = catalog.create_job(collection["id"], config)
            logger.info("Job %s", job_id)
            logger.info("Scanning image files%s", " (forced rehash)" if config["rehash"] else "")
            discover(catalog, job_id, config, previous)
            initial = catalog.files(job_id)
            for row in initial:
                if row["status"] == "error":
                    logger.warning("Scan failed for %r: %s", row["path"], row["error"])
            reused = {
                r["image_id"]
                for r in initial
                if r["status"] != "error"
                and r["image_id"]
                and all(
                    valid_analysis(
                        catalog, catalog.analysis(config["pipeline"], r["image_id"], stage)
                    )
                    for stage in STAGES
                )
            }
            logger.info(
                "Discovered %d paths, %d unique contents; %d fully cached images",
                len(initial),
                len({r["image_id"] for r in initial if r["image_id"]}),
                len(reused),
            )
            reused_stages = {}
            processed = False
            for stage in STAGES:
                records = {
                    r["image_id"]: catalog.analysis(config["pipeline"], r["image_id"], stage)
                    for r in catalog.files(job_id)
                    if r["status"] != "error"
                }
                cached = {
                    image_id
                    for image_id, record in records.items()
                    if valid_analysis(catalog, record)
                }
                reused_stages[stage] = len(cached)
                pending = {
                    image_id
                    for image_id, record in records.items()
                    if image_id not in cached and not attempted(record, job_id)
                }
                if not pending:
                    logger.info("[%s] No pending images (%d cached)", stage, len(cached))
                    continue
                logger.info(
                    "[%s] Starting %d unique images (%d cached)", stage, len(pending), len(cached)
                )
                command = [
                    sys.executable,
                    "-m",
                    "realopen_lens.runtime.worker",
                    "--store",
                    str(store),
                    "--job",
                    job_id,
                    "--stage",
                    stage,
                ]
                processed = True
                stage_runner(catalog, job_id, stage, command, threads=config["threads"])
                logger.info("[%s] Stage finished", stage)
            from realopen_lens.indexing.processing import source_path

            logger.info("Validating source files before publication")
            for row in catalog.files(job_id):
                if row["status"] == "error":
                    continue
                try:
                    if signature(source_path(config, row, check_hash=False)) != json.loads(
                        row["signature"]
                    ):
                        raise ValueError("Source changed during indexing; run index again")
                    issues = image_issues(catalog, config["pipeline"], row["image_id"])
                    with catalog.db:
                        catalog.db.execute(
                            "UPDATE files SET status=?,error=? WHERE job_id=? AND path=?",
                            (
                                "degraded" if issues else "ready",
                                "; ".join(f"{i['stage']}: {i['error']}" for i in issues) or None,
                                job_id,
                                row["path"],
                            ),
                        )
                except (OSError, ValueError) as error:
                    logger.warning("Validation failed for %r: %s", row["path"], error)
                    with catalog.db:
                        catalog.db.execute(
                            "UPDATE files SET status='error',error=? WHERE job_id=? AND path=?",
                            (str(error)[:1500], job_id, row["path"]),
                        )
            files = catalog.files(job_id)
            same = (
                active is not None
                and not processed
                and len(files) == len(previous)
                and all(
                    a["path"] == b["path"]
                    and a["image_id"] == b["image_id"]
                    and a["status"] == b["status"]
                    and a["signature"] == b["signature"]
                    and a["error"] == b["error"]
                    for a, b in zip(files, previous, strict=True)
                )
            )
            if same:
                assert active is not None
                try:
                    same = load_manifest(catalog, active)["pipeline"] == config["pipeline"]
                except (OSError, ValueError):
                    same = False
            logger.info("Keeping published generation" if same else "Publishing search generation")
            generation = (
                active["id"]
                if same and active is not None
                else publish(catalog, job_id, config, collection)
            )
            failed = sum(row["status"] == "error" for row in files)
            degraded = [row for row in files if row["status"] == "degraded"]
            result = IndexResult(
                job_id,
                "completed_with_errors" if failed or degraded else "completed",
                options["collection"],
                len(files),
                len({r["image_id"] for r in files if r["status"] in {"ready", "degraded"}}),
                len(reused),
                failed,
                len({r["path"] for r in previous} - {r["path"] for r in files}),
                generation,
                perf_counter() - started,
                degraded_files=len(degraded),
                degraded_images=len({r["image_id"] for r in degraded}),
                reused_stages=reused_stages,
            )
            catalog.finish(job_id, result.status, result=asdict(result))
            logger.info(
                "%s: %d searchable images, %d failed paths, %d degraded paths, %d reused images, "
                "%d removed paths in %.2fs; generation %s",
                result.status,
                result.images,
                result.failed_files,
                result.degraded_files,
                result.reused_images,
                result.removed_files,
                result.seconds,
                result.generation,
            )
            return result
        except BaseException as error:
            logger.error("Indexing stopped: %s: %s", type(error).__name__, error)
            if job_id:
                catalog.finish(
                    job_id,
                    (
                        "interrupted"
                        if isinstance(error, (KeyboardInterrupt, SystemExit))
                        else "failed"
                    ),
                    error=str(error)[:1500],
                )
            raise
        finally:
            catalog.close()


def index_collection(
    root: Path,
    *,
    collection: str = "personal",
    index_dir: Path | None = None,
    models: Path = Path("models"),
    threads: int = 2,
    languages: str = "eng+fra+ara",
    rehash: bool = False,
    log: bool = False,
    siglip_precision: str = "int8",
    on_progress: Callable[[dict], None] | None = None,
) -> IndexResult:
    if not 1 <= threads <= 8 or not collection.strip() or len(collection) > 128:
        raise ValueError("Use 1–8 threads and a collection name of 1–128 characters")
    if siglip_precision not in {"int8", "fp32"}:
        raise ValueError("SigLIP2 precision must be int8 or fp32")
    store = (index_dir or default_directory()).resolve()
    options = {
        "root": str(root.resolve()),
        "store": str(store),
        "collection": collection,
        "models": str(models.resolve()),
        "threads": threads,
        "languages": languages,
        "rehash": rehash,
        "log": log,
        "siglip_precision": siglip_precision,
    }
    request = store / "requests" / (identifier() + ".json")
    atomic_bytes(request, json.dumps(options).encode())
    process = None
    previous_job = None
    if on_progress is not None:
        try:
            previous_job = (collection_status(collection, store).get("job") or {}).get("id")
        except (OSError, ValueError):
            pass
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "realopen_lens.runtime.worker", "--request", str(request)],
            stdout=subprocess.PIPE,
            stderr=None if log else subprocess.PIPE,
            env=environment(threads),
            start_new_session=sys.platform != "win32",
        )
        if on_progress is None:
            output, errors = process.communicate()
        else:
            while True:
                progress: dict = {"phase": "starting"}
                try:
                    status = collection_status(collection, store)
                    job = status.get("job")
                    if job and job["id"] != previous_job:
                        progress = job
                except (OSError, ValueError):
                    pass
                on_progress(progress)
                try:
                    output, errors = process.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if process.returncode:
            raise RuntimeError(
                (errors or b"").decode(errors="replace")[-2000:]
                or "Index worker failed (see stderr for details)"
            )
        return IndexResult(**json.loads(output))
    except BaseException:
        if process is not None and process.poll() is None:
            if sys.platform != "win32":
                os.kill(process.pid, signal.SIGTERM)
            else:
                stop_worker(process.pid)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_worker(process.pid)
                process.wait(timeout=10)
        raise
    finally:
        request.unlink(missing_ok=True)


def collection_status(collection: str, index_dir: Path | None = None) -> dict:
    catalog = Catalog(index_dir or default_directory(), readonly=True)
    try:
        found = catalog.collection(collection)
        job = catalog.db.execute(
            "SELECT * FROM jobs WHERE collection_id=? ORDER BY started_at DESC LIMIT 1",
            (found["id"],),
        ).fetchone()
        result = {
            "collection": collection,
            "root": found["root"],
            "generation": found["active_generation"],
            "siglip_precision": (
                json.loads(catalog.job(catalog.snapshot(collection)[1]["job_id"])["config"])[
                    "spec"
                ].get("siglip_precision", "int8")
                if found["active_generation"]
                else None
            ),
            "job": None,
        }
        if job:
            files = catalog.files(job["id"])
            config = json.loads(job["config"])
            completed = catalog.db.execute(
                "SELECT COUNT(DISTINCT f.image_id) FROM files f JOIN analyses a "
                "ON f.image_id=a.image_id WHERE f.job_id=? AND a.pipeline=? "
                "AND a.stage=? AND a.status='ok' AND f.status!='error'",
                (job["id"], config["pipeline"], job["phase"]),
            ).fetchone()[0]
            result["job"] = {
                "id": job["id"],
                "status": job["status"],
                "phase": job["phase"],
                "error": job["error"],
                "files": len(files),
                "ready": sum(r["status"] == "ready" for r in files),
                "degraded": sum(r["status"] == "degraded" for r in files),
                "searchable": sum(r["status"] in {"ready", "degraded"} for r in files),
                "current_image": job["current_image"],
                "current_paths": [
                    r["path"]
                    for r in files
                    if job["current_image"] is not None and r["image_id"] == job["current_image"]
                ],
                "phase_completed_images": completed,
                "unique_images": len(
                    {r["image_id"] for r in files if r["image_id"] and r["status"] != "error"}
                ),
                "failures": [
                    {"path": r["path"], "error": r["error"]}
                    for r in files
                    if r["status"] == "error"
                ],
                "degradations": [
                    {"path": r["path"], "error": r["error"]}
                    for r in files
                    if r["status"] == "degraded"
                ],
            }
        return result
    finally:
        catalog.close()
