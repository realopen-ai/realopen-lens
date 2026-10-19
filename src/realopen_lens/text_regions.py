"""Bounded local ONNX text localization; no recognition or implicit downloads."""

import math
from importlib.resources import files
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from realopen_lens.models import digest


def detector_spec() -> dict:
    return json.loads(files("realopen_lens").joinpath("text-detector.lock.json").read_text())


class TextRegionDetector:
    """DB probability-map inference on CPU, with conservative axis-aligned padding.

    Regions are in EXIF-oriented source coordinates. Scores describe detector
    evidence, not calibrated probabilities of text absence or OCR correctness.
    """

    def __init__(self, model: Path, threads: int = 2, max_side: int = 1600):
        if not 1 <= threads <= 32 or not 320 <= max_side <= 2400:
            raise ValueError("Invalid detector thread count or input bound")
        spec = detector_spec()
        if not model.is_file():
            raise ValueError("Missing text detector; run 'lens models fetch-text-detector'")
        if model.stat().st_size != spec["size"] or digest(model) != spec["sha256"]:
            raise ValueError("Text detector integrity check failed")
        try:
            import cv2
        except ImportError as error:
            raise ValueError("Install the 'text-regions' extra for OpenCV") from error
        import onnxruntime as ort

        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.cv = cv2
        self.max_side = max_side
        self.info = {**spec, "threads": threads, "max_side": max_side, "provider": "CPU"}

    def detect(self, image: Image.Image) -> dict:
        if image.mode != "RGB":
            raise ValueError("Text localization requires safely decoded RGB pixels")
        start = perf_counter()
        w, h = image.size
        # Bound the longest edge and preserve narrow panoramas (minimum one stride).
        ratio = min(1.0, self.max_side / max(w, h))
        rw = max(32, int(round(w * ratio / 32)) * 32)
        rh = max(32, int(round(h * ratio / 32)) * 32)
        with image.resize((rw, rh), Image.Resampling.BILINEAR) as resized:
            # The exported detector expects BGR, normalized to [-1, 1].
            pixels = np.asarray(resized, dtype=np.float32)[:, :, ::-1]
            tensor = np.ascontiguousarray((pixels / 127.5 - 1).transpose(2, 0, 1)[None])
        output = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0]
        if output.shape != (1, 1, rh, rw) or not np.isfinite(output).all():
            raise ValueError("Invalid detector probability map")
        regions, saturated = probability_regions(output[0, 0], (w, h), self.cv)
        raw_count = len(regions)
        regions = merge_line_regions(regions)
        return {
            "regions": regions,
            "seconds": perf_counter() - start,
            "input_size": [rw, rh],
            "original_size": [w, h],
            "coordinate_space": "EXIF-oriented source pixels; [left, top, right, bottom]",
            "candidate_limit_reached": saturated,
            "raw_region_count": raw_count,
            "model_sha256": self.info["sha256"],
        }


def probability_regions(probabilities: np.ndarray, size: tuple[int, int], cv) -> tuple[list, bool]:
    """Extract expanded rectangles from DB cores without dropping overflow silently."""
    bitmap = (probabilities > 0.3).astype(np.uint8)
    contours, _ = cv.findContours(bitmap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 2000:
        # Callers preserve the full page rather than selecting arbitrary regions.
        return [], True
    h, w = probabilities.shape
    ow, oh = size
    regions: list[dict] = []
    for contour in contours:
        x, y, cw, ch = cv.boundingRect(contour)
        if cw < 2 or ch < 2:
            continue
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv.drawContours(mask, [contour - [x, y]], -1, 1, thickness=-1)
        score = float(cv.mean(probabilities[y : y + ch, x : x + cw], mask)[0])
        if score < 0.5:
            continue
        # DB cores are shrunken. Expand using area/perimeter distance, keeping
        # rectangular source pixels intact rather than perspective warping glyphs.
        pad = max(2.0, cw * ch * 1.6 / (2 * (cw + ch)))
        box = [
            max(0, math.floor((x - pad) * ow / w)),
            max(0, math.floor((y - pad) * oh / h)),
            min(ow, math.ceil((x + cw + pad) * ow / w)),
            min(oh, math.ceil((y + ch + pad) * oh / h)),
        ]
        regions.append({"box": box, "score": score})
    regions.sort(key=lambda r: (r["box"][1], r["box"][0]))
    return regions, False


def merge_line_regions(regions: list[dict]) -> list[dict]:
    """Keep inter-letter/word gaps when DB fragments a small text line.

    Close rectangles must substantially overlap vertically. Distant columns and
    separate text rows remain separate. The score is the minimum observed member
    score, never an invented OCR confidence.
    """
    merged: list[dict] = []
    for region in sorted(regions, key=lambda r: (r["box"][0], r["box"][1])):
        box = list(region["box"])
        candidate = {**region, "box": box, "members": 1}
        for previous in reversed(merged):
            a, b, c, d = previous["box"]
            x, y, right, bottom = box
            overlap = min(d, bottom) - max(b, y)
            gap = max(0, x - c, a - right)
            if overlap >= min(d - b, bottom - y) * 0.5 and gap <= max(d - b, bottom - y):
                previous["box"] = [min(a, x), min(b, y), max(c, right), max(d, bottom)]
                previous["score"] = min(previous["score"], region["score"])
                previous["members"] += 1
                break
        else:
            merged.append(candidate)
    merged.sort(key=lambda r: (r["box"][1], r["box"][0]))
    return merged


def mask_regions(image: Image.Image, detection: dict) -> Image.Image:
    """Keep source geometry and pixels inside regions; replace only background."""
    if detection["candidate_limit_reached"]:
        return image.copy()
    with image.resize((32, 32)) as sample:
        color = tuple(int(v) for v in np.median(np.asarray(sample).reshape(-1, 3), axis=0))
    with Image.new("L", image.size, 0) as mask:
        drawing = ImageDraw.Draw(mask)
        for region in detection["regions"]:
            left, top, right, bottom = region["box"]
            drawing.rectangle((left, top, right - 1, bottom - 1), fill=255)
        with Image.new("RGB", image.size, color) as background:
            return Image.composite(image, background, mask)
