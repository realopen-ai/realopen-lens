"""Optional RapidOCR experiment; explicit verified local models, never automatic downloads."""

import json
import math
from importlib.metadata import distribution, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from realopen_lens.inference import image_rgb
from realopen_lens.models import digest

LOCK = Path(__file__).parent / "sources" / "rapidocr.lock.json"


def model_info() -> dict:
    spec = json.loads(LOCK.read_text())
    if version("rapidocr") != spec["version"]:
        raise ValueError("Install the pinned ocr-eval dependency group")
    package = Path(str(distribution("rapidocr").locate_file("rapidocr")))
    artifacts = {}
    for name, expected in spec["models"].items():
        path = package / "models" / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"RapidOCR model missing or modified: {name}")
        artifacts[name] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    return {
        "version": spec["version"],
        "artifacts": artifacts,
        "config_sha256": digest(package / "config.yaml"),
    }


def parse_output(result: Any) -> dict:
    texts = list(result.txts) if result.txts is not None else []
    scores = list(result.scores) if result.scores is not None else []
    boxes = list(result.boxes) if result.boxes is not None else []
    if not len(texts) == len(scores) == len(boxes):
        raise ValueError("RapidOCR text/score/box lengths differ")
    lines = []
    for text, score, box in zip(texts, scores, boxes, strict=True):
        polygon = np.asarray(box, dtype=float)
        if not math.isfinite(float(score)) or not 0 <= score <= 1:
            raise ValueError("Invalid RapidOCR confidence")
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            raise ValueError("Invalid RapidOCR polygon")
        lines.append(
            {"text": str(text), "confidence": float(score) * 100, "polygon": polygon.tolist()}
        )
    return {
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "confidence": sum(line["confidence"] for line in lines) / len(lines) if lines else None,
    }


class RapidEngine:
    def __init__(self, threads: int = 2):
        if not 1 <= threads <= 32:
            raise ValueError("OCR requires 1–32 threads")
        from rapidocr import RapidOCR

        self.info = model_info()
        paths = {name: item["path"] for name, item in self.info["artifacts"].items()}
        self.params = {
            "Det.model_path": paths["PP-OCRv6_det_small.onnx"],
            "Rec.model_path": paths["PP-OCRv6_rec_small.onnx"],
            "Cls.model_path": paths["ch_ppocr_mobile_v2.0_cls_mobile.onnx"],
            "Global.max_side_len": 2400,
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.use_coreml": False,
        }
        start = perf_counter()
        self.engine = RapidOCR(params=self.params)
        # Public API warmup loads lazy detector, classifier and recognizer. These
        # fabricated pixels are excluded from quality and per-image latency.
        with Image.new("RGB", (320, 48), "white") as warmup:
            self.engine(warmup)
            self.engine(warmup, use_det=False)
        self.load_seconds = perf_counter() - start

    def extract(self, path: Path) -> dict:
        with image_rgb(path) as image:
            original_size = image.size
            image.thumbnail((2400, 2400))
            result = parse_output(self.engine(image, use_det=True, use_cls=True, use_rec=True))
            result["ocr_size"] = list(image.size)
        warnings = []
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) > 1:
                warnings.append("Only first frame/page processed")
        if not result["text"]:
            warnings.append("No recognized text; absence of text is not established")
        return {
            **result,
            "original_size": list(original_size),
            "warnings": warnings,
            "box_coordinates": "oriented, resized OCR input; line quadrilaterals",
        }
