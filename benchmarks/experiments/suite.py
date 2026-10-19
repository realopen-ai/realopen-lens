"""Resumable per-image evaluation with immutable run identities and retained evidence."""

import argparse
import json
import os
import platform
import shutil
import sys
from time import perf_counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import psutil
from benchmarks.datasets.downloads import digest
from benchmarks.runtime.resources import MemoryMonitor
from benchmarks.runtime.execution import run_stage
from realopen_lens.models.manager import registry, verify
from benchmarks.storage.results import connect, save_stage, reset_errors
from benchmarks.runtime.identity import source_hash
from benchmarks.datasets.queries import query_items
from benchmarks.reporting.images import report
from benchmarks.runtime.resources import write_resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmark-500"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, default=Path("src"))
    parser.add_argument("--variant", choices=["baseline", "improved"], required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--languages", default="eng+fra+ara")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-timeout", type=float, default=90)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--detector-model",
        type=Path,
        help="Experimental Tesseract gating; whole-image is the default",
    )
    args = parser.parse_args()
    if args.detector_model is not None and args.variant != "improved":
        raise ValueError("Text gating requires the current improved pipeline")
    manifest = args.dataset / "manifest.json"
    rows = json.loads(manifest.read_text())["images"]
    if args.image_timeout <= 0 or args.startup_timeout <= 0:
        raise ValueError("Worker timeouts must be positive")
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            raise ValueError("Limit must select at least one available image")
        rows = rows[: args.limit]
    for row in rows:
        path = (args.dataset / row["path"]).resolve()
        if not path.is_relative_to(args.dataset.resolve()) or digest(path) != row["sha256"]:
            raise ValueError(f"Invalid dataset input: {row['id']}")
    verify(args.models, "int8")
    identity = {
        "dataset": str(args.dataset.resolve()),
        "manifest_sha256": digest(manifest),
        "images": rows,
        "queries": [
            q
            for q in query_items(rows)
            if q["id"] not in set(json.loads(manifest.read_text()).get("excluded_queries", []))
        ],
        "engine_root": str(args.engine_root.resolve()),
        "engine_sha256": source_hash(args.engine_root),
        "variant": args.variant,
        "models": str(args.models.resolve()),
        "artifacts": registry(),
        "threads": args.threads,
        "languages": args.languages,
        "worker_limits": {
            "image_timeout": args.image_timeout,
            "startup_timeout": args.startup_timeout,
        },
        "runner_sha256": source_hash(Path("benchmarks")),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "ram_gib": psutil.virtual_memory().total / 2**30,
            "python": sys.version,
        },
        "packages": {
            name: version(name)
            for name in ["numpy", "Pillow", "onnxruntime", "tokenizers", "psutil", "realopen-lens"]
        },
    }
    if args.detector_model is not None:
        from realopen_lens.ocr.regions import detector_spec

        spec = detector_spec()
        if (
            args.detector_model.stat().st_size != spec["size"]
            or digest(args.detector_model) != spec["sha256"]
        ):
            raise ValueError("Text detector integrity check failed")
        identity["text_detector"] = {
            "path": str(args.detector_model.resolve()),
            "sha256": spec["sha256"],
            "mode": "gate",
        }
    try:
        identity["packages"]["opencv-python-headless"] = version("opencv-python-headless")
    except PackageNotFoundError:
        if args.detector_model is not None:
            raise ValueError("Install the text-regions extra to evaluate gating") from None
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError(
            "Resume configuration/source/environment mismatch; choose a new output directory"
        )
    config_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")
    for source, name in (
        (args.engine_root, "engine-source"),
        (Path("benchmarks"), "harness-source"),
    ):
        if not (args.output / name).exists():
            shutil.copytree(
                source, args.output / name, ignore=shutil.ignore_patterns("__pycache__")
            )
    db = connect(args.output / "results.sqlite")
    if args.retry_errors:
        reset_errors(db)
    db.close()
    memory = MemoryMonitor()
    try:
        with memory:
            for stage in ["ocr", "vision", "semantic", "visual_queries", "semantic_queries"]:
                if source_hash(args.engine_root) != identity["engine_sha256"]:
                    raise ValueError(
                        "Engine changed during run; resume with preserved original source"
                    )
                memory.phase = stage
                print(f"Running {stage} (completed records are reused)", flush=True)
                env = {
                    **os.environ,
                    "PYTHONPATH": str(args.engine_root.resolve()),
                    "OMP_NUM_THREADS": str(args.threads),
                    "OPENBLAS_NUM_THREADS": str(args.threads),
                    "TOKENIZERS_PARALLELISM": "false",
                }
                stage_start = perf_counter()
                stage_db = connect(args.output / "results.sqlite")
                try:
                    events = run_stage(
                        [
                            sys.executable,
                            "-m",
                            "benchmarks.runtime.suite_worker",
                            str(config_path.resolve()),
                            stage,
                        ],
                        stage_db,
                        stage,
                        env,
                        args.output / f"{stage}.log",
                        lambda image_id, payload: save_stage(stage_db, image_id, stage, payload),
                        image_timeout=args.image_timeout,
                        startup_timeout=args.startup_timeout,
                        stage_timeout=max(600, len(rows) * 60),
                    )
                finally:
                    stage_db.close()
                with (args.output / "run-events.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "stage": stage,
                                "seconds": perf_counter() - stage_start,
                                "returncode": 0,
                                "supervision": events,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                        )
                        + "\n"
                    )
            memory.phase = "evaluation"
            report(args.output, identity)
    finally:
        write_resources(args.output, memory)


if __name__ == "__main__":
    main()
