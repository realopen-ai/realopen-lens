"""Dependency-scoped cache aliases over the existing durable analysis store."""

import hashlib
import json
from pathlib import Path
from typing import Any

from realopen_lens.models.manager import digest
from realopen_lens.storage.artifacts import valid_analysis
from realopen_lens.storage.catalog import Catalog, STAGES


def implementation(path: Path) -> str:
    """Hash shipped source in both Python and frozen builds; fail closed if absent."""
    source = path.with_suffix(".py") if path.suffix in {".pyc", ".pyo"} else path
    if not source.is_file():
        # Unknown packaging must not pretend that bytecode equals source.
        return digest(path)
    return digest(source)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def dependencies(spec: dict, stage: str) -> dict:
    """Shared processing/preprocessing changes invalidate conservatively."""
    common = {k: spec[k] for k in ("processing_sha256", "inference_sha256")}
    packages = spec["packages"]
    selected = spec.get("siglip_precision", "int8")
    models = spec["model_registry"]["models"]
    if stage == "metadata":
        return {**common, "Pillow": packages["Pillow"], "thumbnail": spec["thumbnail_side"]}
    if stage == "ocr":
        return {
            **common,
            "packages": {k: packages[k] for k in ("numpy", "Pillow")},
            **{
                k: spec[k]
                for k in ("ocr_sha256", "ocr_environment", "languages", "threads", "ocr_characters")
            },
        }
    base = {**common, "threads": spec["threads"]}
    if stage == "vision":
        return {
            **base,
            "packages": {k: packages[k] for k in ("numpy", "Pillow", "onnxruntime")},
            "graph": models["siglip2"]["files"][f"vision_{selected}"],
        }
    if stage == "semantic":
        return {
            **base,
            "packages": {k: packages[k] for k in ("numpy", "onnxruntime", "tokenizers")},
            "graph": models["e5"]["files"]["text_int8"],
            "tokenizer": models["e5"]["files"]["tokenizer"],
            "ocr": dependencies(spec, "ocr"),
            "characters": spec["semantic_characters"],
            "passages": spec["semantic_passages"],
        }
    raise ValueError("Unknown indexing stage")


def key(catalog: Catalog, config: dict, stage: str, image_id: str) -> str | None:
    value: dict = {
        "cache_schema": 1,
        "stage": stage,
        "dependencies": dependencies(config["spec"], stage),
    }
    if stage == "semantic":
        ocr = catalog.analysis(config["pipeline"], image_id, "ocr")
        if not valid_analysis(catalog, ocr):
            return None
        assert ocr is not None
        value["ocr_text"] = json.loads(ocr["payload"])["text"][
            : config["spec"]["semantic_characters"]
        ]
    return fingerprint(value)


def compatible(catalog: Catalog, config: dict, stage: str) -> list[str]:
    """Legacy rows are reused only when their recorded dependencies match exactly."""
    expected = dependencies(config["spec"], stage)
    from realopen_lens.encoders import onnx
    from realopen_lens.indexing import processing
    from realopen_lens.ocr import tesseract

    identities = {}
    for name, module in (
        ("inference_sha256", onnx),
        ("processing_sha256", processing),
        ("ocr_sha256", tesseract),
    ):
        if module.__file__ is None:
            raise ValueError("Stage implementation provenance is unavailable")
        path = Path(module.__file__)
        identities[name] = (digest(path), implementation(path))
    matches = set()
    for row in catalog.db.execute("SELECT DISTINCT config FROM jobs"):
        old = json.loads(row[0])
        try:
            spec = dict(old["spec"])
            for name, (raw, stable) in identities.items():
                if spec.get(name) == raw:
                    spec[name] = stable
            if dependencies(spec, stage) == expected:
                matches.add(old["pipeline"])
        except (KeyError, TypeError):
            continue  # Unrecognized old configuration cannot establish equivalence.
    return sorted(matches - {config["pipeline"]})


def alias(catalog: Catalog, pipeline: str, image_id: str, stage: str, record: dict):
    catalog.save_analysis(
        pipeline,
        image_id,
        stage,
        json.loads(record["payload"]),
        record["artifact"],
        record["checksum"],
        job_id=record.get("job_id"),
    )


def restore(catalog: Catalog, job_id: str, config: dict, stage: str) -> None:
    pipelines = compatible(catalog, config, stage)
    ids = {r["image_id"] for r in catalog.files(job_id) if r["status"] != "error"}
    for image_id in ids:
        namespace = key(catalog, config, stage, image_id)
        if namespace is None:
            continue
        current = catalog.analysis(config["pipeline"], image_id, stage)
        if valid_analysis(catalog, current):
            continue
        cached = catalog.analysis(namespace, image_id, stage)
        if not valid_analysis(catalog, cached):
            cached = None
            for pipeline in pipelines:
                candidate = catalog.analysis(pipeline, image_id, stage)
                if not valid_analysis(catalog, candidate):
                    continue
                if stage == "semantic":
                    old_ocr = catalog.analysis(pipeline, image_id, "ocr")
                    new_ocr = catalog.analysis(config["pipeline"], image_id, "ocr")
                    if not valid_analysis(catalog, old_ocr) or not new_ocr or not old_ocr:
                        continue
                    if (
                        json.loads(old_ocr["payload"])["text"]
                        != json.loads(new_ocr["payload"])["text"]
                    ):
                        continue
                cached = candidate
                break
        if cached is not None:
            alias(catalog, config["pipeline"], image_id, stage, cached)
        elif current is not None and current["status"] == "ok":
            # A corrupt artifact is pending work, not a completed cache hit.
            with catalog.db:
                catalog.db.execute(
                    "DELETE FROM analyses WHERE pipeline=? AND image_id=? AND stage=?",
                    (config["pipeline"], image_id, stage),
                )


def remember(catalog: Catalog, job_id: str, config: dict, stage: str) -> None:
    for image_id in {r["image_id"] for r in catalog.files(job_id) if r["status"] != "error"}:
        namespace = key(catalog, config, stage, image_id)
        record = catalog.analysis(config["pipeline"], image_id, stage)
        if namespace and valid_analysis(catalog, record):
            assert record is not None
            existing = catalog.analysis(namespace, image_id, stage)
            if existing and all(
                existing[k] == record[k] for k in ("status", "payload", "artifact", "checksum")
            ):
                continue
            alias(catalog, namespace, image_id, stage, record)


def prime(catalog: Catalog, job_id: str, config: dict) -> None:
    for stage in STAGES:
        restore(catalog, job_id, config, stage)
