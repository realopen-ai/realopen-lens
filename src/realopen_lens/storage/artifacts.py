"""Atomic derived artifacts and immutable, disk-backed search generations."""

import json
import os
from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4

import numpy as np

from realopen_lens.storage.catalog import Catalog, REQUIRED_STAGES, identifier, local_path
from realopen_lens.models.manager import digest
from realopen_lens.retrieval.ranking import lexical_index
from realopen_lens.ocr.languages import selection


def atomic_bytes(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_vectors(path: Path, vector: np.ndarray):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 2 or not np.isfinite(vector).all():
        raise ValueError("Invalid embedding matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        with temporary.open("wb") as stream:
            np.save(stream, vector, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_analysis(catalog: Catalog, record: dict | None) -> bool:
    if not record or record["status"] != "ok":
        return False
    if record["artifact"]:
        path = local_path(catalog.root, record["artifact"])
        return path.is_file() and digest(path) == record["checksum"]
    return True


def attempted(record: dict | None, job_id: str) -> bool:
    """Retry failed stages in new jobs, never repeatedly in a restarted worker."""
    return bool(
        record and record.get("job_id") == job_id and record["status"] in {"error", "skipped"}
    )


def image_issues(catalog: Catalog, pipeline: str, image_id: str) -> list[dict]:
    for stage in REQUIRED_STAGES:
        if not valid_analysis(catalog, catalog.analysis(pipeline, image_id, stage)):
            raise ValueError(f"Required {stage} stage is incomplete or corrupt")
    issues = []
    for stage in ("ocr", "semantic"):
        record = catalog.analysis(pipeline, image_id, stage)
        if (
            stage == "ocr"
            and record is not None
            and valid_analysis(catalog, record)
            and json.loads(record["payload"]).get("timeout_recovery")
        ):
            issues.append(
                {
                    "stage": "ocr",
                    "error_type": "PartialOCR",
                    "error": "Whole-image OCR timed out; recovered band text is incomplete",
                }
            )
        if not valid_analysis(catalog, record):
            payload = json.loads(record["payload"]) if record else {}
            issues.append(
                {
                    "stage": stage,
                    "error_type": payload.get("error_type", "IncompleteStage"),
                    "error": payload.get("error", "Stage is incomplete or corrupt"),
                }
            )
            if stage == "ocr":
                break  # Semantic OCR retrieval depends on successful OCR.
    return issues


def load_manifest(catalog: Catalog, generation: dict) -> dict:
    path = local_path(catalog.root, generation["manifest"])
    if not path.is_file() or digest(path) != generation["checksum"]:
        raise ValueError("Index manifest is missing or corrupt; run 'lens index' to rebuild")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Unsupported index manifest version; run 'lens index' to rebuild")
    for artifact in manifest["artifacts"].values():
        path = local_path(catalog.root, artifact["path"])
        if not path.is_file() or digest(path) != artifact["sha256"]:
            raise ValueError("Index artifact is missing or corrupt; run 'lens index' to rebuild")
    return manifest


def matrices(catalog: Catalog, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for name, ids, dimension in [("vision", "vision_ids", 768), ("semantic", "passage_ids", 384)]:
        path = local_path(catalog.root, manifest["artifacts"][name]["path"])
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(manifest[ids]), dimension) or array.dtype != np.float32:
            raise ValueError("Embedding matrix does not match its published mapping")
        arrays.append(array)
    return arrays[0], arrays[1]


def publish(catalog: Catalog, job_id: str, config: dict, collection: dict) -> str:
    files = catalog.files(job_id)
    paths_by_id: dict[str, list[dict]] = {}
    for row in files:
        if row["status"] in {"ready", "degraded"}:
            paths_by_id.setdefault(row["image_id"], []).append(row)
    image_ids = sorted(paths_by_id)
    generation_id = identifier()
    base = Path("generations") / generation_id
    directory = catalog.root / base
    building = directory.with_name(directory.name + ".building")
    records, passage_ids = [], []
    analyses, issues_by_id = {}, {}
    for image_id in image_ids:
        issues_by_id[image_id] = image_issues(catalog, config["pipeline"], image_id)
        stages: dict[str, dict] = {}
        for stage in REQUIRED_STAGES:
            record = catalog.analysis(config["pipeline"], image_id, stage)
            if record is None or not valid_analysis(catalog, record):
                raise ValueError("Analysis changed before index publication")
            stages[stage] = record
        analyses[image_id] = stages
        ocr_record = catalog.analysis(config["pipeline"], image_id, "ocr")
        ocr = (
            json.loads(ocr_record["payload"])
            if ocr_record and valid_analysis(catalog, ocr_record)
            else {}
        )
        records.append(
            {
                "language_selection": selection(ocr, config["spec"].get("languages")),
                "image_id": image_id,
                "text": ocr.get("text", ""),
                "alternate_texts": ocr.get("alternate_texts", []),
            }
        )
        semantic = catalog.analysis(config["pipeline"], image_id, "semantic")
        if ocr and semantic and valid_analysis(catalog, semantic):
            stages["semantic"] = semantic
            passage_ids.extend([image_id] * json.loads(semantic["payload"])["passages"])
    try:
        building.mkdir(parents=True)
        for name, ids, dimension in [("vision", image_ids, 768), ("semantic", passage_ids, 384)]:
            path = building / (name + ".npy")
            if ids:
                output = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(len(ids), dimension)
                )
                offset = 0
                for image_id in image_ids:
                    record = analyses[image_id].get(name)
                    if record and record["artifact"]:
                        array = np.load(
                            local_path(catalog.root, record["artifact"]), allow_pickle=False
                        )
                        if array.ndim != 2 or array.shape[1] != dimension:
                            raise ValueError("Invalid cached embedding dimensions")
                        output[offset : offset + len(array)] = array
                        offset += len(array)
                if offset != len(ids):
                    raise ValueError("Embedding row mapping mismatch")
                output.flush()
                del output
            else:
                save_vectors(path, np.empty((0, dimension), dtype=np.float32))
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
        index = lexical_index(records)
        index.commit()
        disk = sqlite3.connect(building / "lexical.sqlite3")
        try:
            index.backup(disk)
        finally:
            disk.close()
            index.close()
        with (building / "lexical.sqlite3").open("r+b") as stream:
            os.fsync(stream.fileno())
        members = []
        for image_id in image_ids:
            data = json.loads(analyses[image_id]["metadata"]["payload"])
            members.append(
                {
                    "image_id": image_id,
                    "paths": [r["path"] for r in paths_by_id[image_id]],
                    "signatures": {
                        r["path"]: json.loads(r["signature"]) for r in paths_by_id[image_id]
                    },
                    "metadata": data,
                    "thumbnail": analyses[image_id]["metadata"]["artifact"],
                    "thumbnail_sha256": analyses[image_id]["metadata"]["checksum"],
                    "status": "degraded" if issues_by_id[image_id] else "ready",
                    "issues": issues_by_id[image_id],
                }
            )
        artifacts = {
            name: {"path": (base / filename).as_posix(), "sha256": digest(building / filename)}
            for name, filename in [
                ("vision", "vision.npy"),
                ("semantic", "semantic.npy"),
                ("lexical", "lexical.sqlite3"),
            ]
        }
        manifest = {
            "version": 1,
            "pipeline": config["pipeline"],
            "config": config,
            "vision_ids": image_ids,
            "passage_ids": passage_ids,
            "images": members,
            "ocr": records,
            "artifacts": artifacts,
        }
        path = building / "manifest.json"
        atomic_bytes(path, (json.dumps(manifest, ensure_ascii=False) + "\n").encode())
        os.replace(building, directory)
        catalog.publish(
            collection["id"],
            job_id,
            generation_id,
            (base / "manifest.json").as_posix(),
            digest(directory / "manifest.json"),
        )
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return generation_id
