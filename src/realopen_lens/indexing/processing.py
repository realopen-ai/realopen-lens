"""Bounded single-model indexing stages, with durable per-image completion."""

import io
import json
import logging
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

import numpy as np
from PIL import Image

from realopen_lens.storage.catalog import Catalog, local_path
from realopen_lens.storage.artifacts import atomic_bytes, attempted, save_vectors, valid_analysis
from realopen_lens.models.manager import artifact_path, digest

logger = logging.getLogger(__name__)


def source_path(config: dict, row: dict, *, check_hash: bool = True) -> Path:
    root = Path(config["root"])
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Source file disappeared or escaped the collection root")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Image exceeds 64 MiB encoded size")
    if check_hash and digest(path) != row["image_id"]:
        raise ValueError("Source changed during indexing; run index again")
    return path


def encoder(stage: str, config: dict):
    from realopen_lens.encoders.onnx import TextEncoder, VisionEncoder

    root = Path(config["models"])
    if stage == "vision":
        precision = config.get("siglip_precision", "int8")
        return VisionEncoder(
            artifact_path(root, "siglip2", f"vision_{precision}"), config["threads"]
        )
    return TextEncoder(
        artifact_path(root, "e5", "text_int8"),
        artifact_path(root, "e5", "tokenizer"),
        "e5",
        config["threads"],
    )


def metadata(path: Path) -> tuple[dict, bytes]:
    from realopen_lens.encoders.onnx import image_rgb

    with Image.open(path) as source:
        result: dict[str, Any] = {
            "format": source.format,
            "source_size": list(source.size),
            "frames": getattr(source, "n_frames", 1),
            "bytes": path.stat().st_size,
        }
    with image_rgb(path) as image:
        result.update(width=image.width, height=image.height)
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
    result["warnings"] = ["Only first frame/page indexed"] if result["frames"] > 1 else []
    return result, buffer.getvalue()


def compact_ocr(result: dict) -> dict:
    data = {k: v for k, v in result.items() if k not in ("attempts", "alternate_texts")}
    data["warnings"] = list(data.get("warnings", []))
    if len(data["text"]) > 128000 or len(data.get("words", [])) > 20000:
        data["warnings"].append("OCR storage budget reached; text/word details truncated")
    data["text"] = data["text"][:128000]
    data["words"] = data.get("words", [])[:20000]
    alternatives = result.get("alternate_texts", [])
    data["alternate_texts"] = [text[:32000] for text in alternatives[:3]]
    if len(alternatives) > 3 or any(len(t) > 32000 for t in alternatives):
        data["warnings"].append("Alternate OCR storage budget reached")
    data["attempts"] = [
        {
            k: v
            for k, v in attempt.items()
            if k
            in ("psm", "input", "seconds", "confidence", "ocr_size", "source_region", "rotation")
        }
        for attempt in result.get("attempts", [])
    ]
    return data


def process_stage(catalog: Catalog, job_id: str, stage: str):
    config = json.loads(catalog.job(job_id)["config"])
    pending = {}
    for row in catalog.files(job_id):
        if row["status"] == "error" or row["image_id"] in pending:
            continue
        cached = catalog.analysis(config["pipeline"], row["image_id"], stage)
        if valid_analysis(catalog, cached) or attempted(cached, job_id):
            continue
        if stage == "semantic" and not valid_analysis(
            catalog, catalog.analysis(config["pipeline"], row["image_id"], "ocr")
        ):
            catalog.save_analysis(
                config["pipeline"],
                row["image_id"],
                stage,
                {"status": "skipped", "reason": "OCR unavailable", "passages": 0},
                job_id=job_id,
            )
            logger.info("[semantic] Skipped %r: OCR unavailable", row["path"])
            continue
        pending[row["image_id"]] = row
    if not pending:
        return
    if stage in ("vision", "semantic"):
        logger.info("[%s] Loading local encoder (%d CPU threads)", stage, config["threads"])
    model = encoder(stage, config) if stage in ("vision", "semantic") else None
    for position, (image_id, row) in enumerate(pending.items(), 1):
        logger.info("[%s %d/%d] Processing %r", stage, position, len(pending), row["path"])
        catalog.heartbeat(job_id, stage, image_id, monotonic())
        started = perf_counter()
        artifact = None
        checksum = None
        try:
            path = source_path(config, row)
            relative = (
                Path("analyses")
                / config["pipeline"]
                / image_id
                / ("thumbnail.jpg" if stage == "metadata" else stage + ".npy")
            )
            target = local_path(catalog.root, relative.as_posix())
            if stage == "metadata":
                data, thumbnail = metadata(path)
                source_path(config, row)
                atomic_bytes(target, thumbnail)
                artifact, checksum = relative.as_posix(), digest(target)
            elif stage == "ocr":
                from realopen_lens.ocr.tesseract import extract

                data = compact_ocr(extract(path, config["languages"], config["threads"]))
            else:
                assert model is not None
                if stage == "vision":
                    vector = model.encode([path])
                    data = {"vectors": 1}
                else:
                    found = catalog.analysis(config["pipeline"], image_id, "ocr")
                    if found is None or found["status"] != "ok":
                        raise ValueError("OCR unavailable for semantic embedding")
                    text = json.loads(found["payload"])["text"]
                    passages = model.passages(text[:32000]) if text.strip() else []
                    data = {"passages": min(len(passages), 16), "warnings": []}
                    if len(text) > 32000 or len(passages) > 16:
                        data["warnings"].append(
                            "Semantic budget reached; first 16 passages / 32000 characters"
                        )
                    vector = model.encode(passages[:16], passage=True) if passages else None
                if vector is not None:
                    dimension = 768 if stage == "vision" else 384
                    expected_rows = 1 if stage == "vision" else data["passages"]
                    vector = np.asarray(vector, dtype=np.float32)
                    if (
                        vector.shape != (expected_rows, dimension)
                        or not np.isfinite(vector).all()
                        or not np.allclose(np.linalg.norm(vector, axis=1), 1, atol=1e-3)
                    ):
                        raise ValueError("Invalid or non-normalized encoder vectors")
                    source_path(config, row)
                    save_vectors(target, vector)
                    artifact, checksum = relative.as_posix(), digest(target)
            source_path(config, row)
            data["status"] = "ok"
        except Exception as error:
            data = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error)[:1500],
            }
            artifact, checksum = None, None
            try:
                source_path(config, row)
            except Exception:
                catalog.fail_image(job_id, image_id, f"{stage}: {error}")
            else:
                catalog.fail_stage(job_id, image_id, stage, str(error))
        data["seconds"] = perf_counter() - started
        catalog.save_analysis(
            config["pipeline"], image_id, stage, data, artifact, checksum, job_id=job_id
        )
        if data["status"] == "ok":
            logger.info(
                "[%s %d/%d] Completed %r in %.2fs",
                stage,
                position,
                len(pending),
                row["path"],
                data["seconds"],
            )
            for warning in data.get("warnings", []):
                logger.warning("[%s] %r: %s", stage, row["path"], warning)
        else:
            logger.warning(
                "[%s %d/%d] Stage failed for %r in %.2fs: %s",
                stage,
                position,
                len(pending),
                row["path"],
                data["seconds"],
                data["error"],
            )
