"""Dependency-scoped cache aliases over the existing durable analysis store."""

import ast
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


# Exact reviewed pre-EXIF worker source: only metadata() changed. Unknown older
# workers retain their own identity and fail closed instead of assuming equivalence.
_LEGACY_WORKER_SOURCE = "d7878c33d6343fdfd600f4c56311bdb7947f1bba5aaa64916e18329a337e14ce"
_LEGACY_WORKER_SHARED = "364ec9d07fe83cd1c15bd341d1ad8a79812014e552d0edc6c9f3d7c54b94b1ae"
_LEGACY_PROCESSING = {_LEGACY_WORKER_SOURCE: _LEGACY_WORKER_SHARED}


def worker_identity(path: Path) -> dict:
    source = path.with_suffix(".py") if path.suffix in {".pyc", ".pyo"} else path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    metadata: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "metadata"
    ]
    tree.body = [node for node in tree.body if node not in metadata]
    return {
        "source": implementation(path),
        "shared": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        "metadata": hashlib.sha256(
            ast.dump(ast.Module(body=metadata, type_ignores=[]), include_attributes=False).encode()
        ).hexdigest(),
    }


def shared_worker(spec: dict) -> str:
    identity = spec.get("processing_identity", {})
    source = spec["processing_sha256"]
    if identity.get("source") == source:
        return identity["shared"]
    return _LEGACY_PROCESSING.get(source, source)


# HEIF is a newly accepted container; the pixel/embedding preprocessing of all
# previously accepted formats is identical. Normalize only this reviewed guard.
_LEGACY_ENCODER_SOURCE = "5de67f88fe86c68bb8f6b20194482d7be6046570e54463017340bd7c73045e6d"
_LEGACY_ENCODER_SHARED = "8e936f42ff8ae683c5307618c11b8e1dfaef1277690da2354cac84f6abea1d69"
_LEGACY_ENCODERS = {_LEGACY_ENCODER_SOURCE: _LEGACY_ENCODER_SHARED}


def encoder_identity(path: Path) -> str:
    source = path.with_suffix(".py") if path.suffix in {".pyc", ".pyo"} else path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and isinstance(node.left.value, ast.Name)
            and node.left.value.id == "source"
            and node.left.attr == "format"
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Set)
        ):
            node.comparators[0].elts = [
                item
                for item in node.comparators[0].elts
                if not (isinstance(item, ast.Constant) and item.value == "HEIF")
            ]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def normalize_encoder(value: str) -> str:
    return _LEGACY_ENCODERS.get(value, value)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def dependencies(spec: dict, stage: str) -> dict:
    """Shared processing/preprocessing changes invalidate conservatively."""
    common = {
        "processing": shared_worker(spec),
        "inference_sha256": normalize_encoder(spec["inference_sha256"]),
    }
    packages = spec["packages"]
    selected = spec.get("siglip_precision", "int8")
    models = spec["model_registry"]["models"]
    if stage == "metadata":
        return {
            **common,
            "Pillow": packages["Pillow"],
            "thumbnail": spec["thumbnail_side"],
            "metadata_implementation": spec.get("metadata_sha256"),
            "metadata_worker": spec.get("processing_identity", {}).get("metadata"),
            "heif_decoder": spec.get("heif_decoder"),
        }
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
    metadata = catalog.analysis(config["pipeline"], image_id, "metadata")
    if stage != "metadata":
        # Until metadata is restored/recomputed, the container decoder dependency
        # is unknown. Wait for the required metadata stage before reusing analyses.
        if not valid_analysis(catalog, metadata):
            return None
        assert metadata is not None
        if json.loads(metadata["payload"]).get("format") in {"HEIF", "HEIC"}:
            value["heif_decoder"] = config["spec"].get("heif_decoder")
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
                assert candidate is not None
                metadata = catalog.analysis(config["pipeline"], image_id, "metadata")
                if (
                    stage != "metadata"
                    and metadata
                    and metadata["status"] == "ok"
                    and json.loads(metadata["payload"]).get("format") in {"HEIF", "HEIC"}
                ):
                    producer = candidate.get("job_id")
                    if not producer:
                        continue
                    previous = json.loads(catalog.job(producer)["config"])["spec"]
                    if previous.get("heif_decoder") != config["spec"].get("heif_decoder"):
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
