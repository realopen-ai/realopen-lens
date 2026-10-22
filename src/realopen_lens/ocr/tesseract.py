"""Local Tesseract OCR with bounded retries and explicit low-quality diagnostics."""

import io
from contextlib import ExitStack, contextmanager
import math
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from realopen_lens.encoders.onnx import image_rgb
from realopen_lens.models.manager import digest
from realopen_lens.ocr import runtime


@lru_cache(maxsize=8)
def _environment(executable: str, languages: str) -> dict[str, Any]:
    version = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env=runtime.environment(executable),
    )
    result = subprocess.run(
        [executable, "--list-langs"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env=runtime.environment(executable),
    )
    requested = languages.split("+")
    if not all(re.fullmatch(r"[A-Za-z0-9_]+", code) for code in requested):
        raise ValueError("Invalid Tesseract language codes")
    missing = set(requested) - set(result.stdout.splitlines()[1:])
    if missing:
        raise ValueError(f"Missing Tesseract language packs: {sorted(missing)}")
    match = re.search(r'in "([^"]+)"', result.stdout)
    hashes = {}
    if match:
        for code in requested:
            path = Path(match[1]) / f"{code}.traineddata"
            if path.is_file():
                hashes[code] = digest(path)
    provenance: dict[str, Any] = {
        "version": version.stdout.splitlines()[0],
        "language_pack_sha256": hashes,
    }
    bundled = runtime.fingerprint(executable)
    if bundled:
        provenance["bundled_runtime_sha256"] = bundled
    return provenance


def parse_tsv(text: str) -> dict[str, Any]:
    """Retain reading-order lines, confidences, and boxes in OCR-input coordinates."""
    words: list[dict[str, Any]] = []
    lines: dict[tuple[int, ...], list[str]] = {}
    rows = text.splitlines()
    if not rows or not rows[0].startswith("level\tpage_num\tblock_num"):
        raise ValueError("Invalid Tesseract TSV header")
    for row in rows[1:]:
        cells = row.split("\t", 11)
        if cells[0] != "5":
            continue
        if len(cells) != 12:
            raise ValueError("Malformed Tesseract word row")
        if not cells[11].strip():
            continue
        confidence = float(cells[10])
        if not math.isfinite(confidence):
            raise ValueError("Invalid Tesseract word confidence")
        if confidence < 0:
            continue
        if not 0 <= confidence <= 100:
            raise ValueError("Invalid Tesseract word confidence")
        key = tuple(int(value) for value in cells[1:5])
        token = cells[11].strip()
        box = [int(value) for value in cells[6:10]]
        if min(box) < 0 or box[2] == 0 or box[3] == 0:
            raise ValueError("Invalid Tesseract word box")
        lines.setdefault(key, []).append(token)
        words.append({"text": token, "confidence": confidence, "box": box})
    return {
        "text": "\n".join(" ".join(line) for line in lines.values()),
        "words": words,
        "confidence": sum(w["confidence"] for w in words) / len(words) if words else None,
    }


def _recognize(
    image: Image.Image, executable: str, languages: str, threads: int, psm: int, timeout: float
) -> dict[str, Any]:
    start = monotonic()
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    remaining = timeout - (monotonic() - start)
    if remaining <= 0:
        raise subprocess.TimeoutExpired("OCR image serialization", timeout)
    result = subprocess.run(
        [executable, "stdin", "stdout", "-l", languages, "--oem", "1", "--psm", str(psm), "tsv"],
        input=payload.getvalue(),
        capture_output=True,
        timeout=remaining,
        env={**runtime.environment(executable), "OMP_THREAD_LIMIT": str(threads)},
    )
    if result.returncode:
        raise ValueError(
            f"Tesseract failed ({result.returncode}): "
            f'{result.stderr.decode(errors="replace")[:500]}'
        )
    if len(result.stdout) > 8 * 1024 * 1024:
        raise ValueError("Tesseract output exceeded 8 MiB")
    return {
        **parse_tsv(result.stdout.decode("utf-8")),
        "psm": psm,
        "ocr_size": list(image.size),
        "seconds": monotonic() - start,
        "stderr": result.stderr.decode(errors="replace")[:1000],
    }


def _quality(result: dict[str, Any]) -> float:
    # Prefer the amount of readable evidence, rather than rewarding one perfect word.
    return sum(len(word["text"]) * max(0, word["confidence"] - 20) for word in result["words"])


def _layout_incomplete(image: Image.Image, words: list[dict[str, Any]]) -> bool:
    """Notice substantial ink below the recognized text, as in boxed diagrams."""
    if not words:
        return True
    bottom = max(word["box"][1] + word["box"][3] for word in words)
    if bottom >= image.height * 0.75:
        return False
    pixels = np.asarray(image)
    ink_below = np.count_nonzero(pixels[min(image.height, bottom + 10) :] < 160)
    return bool(ink_below > image.width * image.height * 0.002)


def _remove_rules(image: Image.Image) -> tuple[Image.Image, bool]:
    """Remove only narrow, long ruling lines from an OCR fallback copy."""
    pixels = np.array(image)
    dark = pixels < 160
    changed = False
    for axis, threshold in ((1, 0.6), (0, 0.25)):
        positions = np.flatnonzero(dark.mean(axis=axis) > threshold)
        groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        for group in groups:
            if not 1 <= len(group) <= 5:
                continue
            start, stop = max(0, int(group[0]) - 1), int(group[-1]) + 2
            if axis == 1:
                pixels[start:stop, :] = 255
            else:
                pixels[:, start:stop] = 255
            changed = True
    return Image.fromarray(pixels), changed


def _orientation(image: Image.Image, executable: str, threads: int, timeout: float) -> dict:
    environment = _environment(executable, "osd")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    result = subprocess.run(
        [executable, "stdin", "stdout", "-l", "osd", "--psm", "0"],
        input=payload.getvalue(),
        capture_output=True,
        timeout=timeout,
        env={**runtime.environment(executable), "OMP_THREAD_LIMIT": str(threads)},
    )
    text = result.stdout.decode(errors="replace")
    angle = re.search(r"Rotate: (\d+)", text)
    confidence = re.search(r"Orientation confidence: ([\d.]+)", text)
    return {
        "rotation": int(angle[1]) if angle else 0,
        "confidence": float(confidence[1]) if confidence else 0,
        "stderr": result.stderr.decode(errors="replace")[:1000],
        "language_pack_sha256": environment["language_pack_sha256"],
    }


def _gray(image: Image.Image) -> tuple[Image.Image, bool]:
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    inverted = float(np.median(np.asarray(gray))) < 128
    if inverted:
        gray = ImageOps.invert(gray)
    return gray, inverted


def _source_region(path: Path, size: tuple[int, int]) -> tuple[int, int, int, int]:
    full = (0, 0, *size)
    with Image.open(path) as source:
        if "A" not in source.getbands():
            return full
        oriented = ImageOps.exif_transpose(source)
        try:
            alpha = oriented.getchannel("A")
            try:
                mask = alpha.point([0] * 128 + [255] * 128)
                bbox = mask.getbbox()
                mask.close()
            finally:
                alpha.close()
        finally:
            oriented.close()
    # Treat only a substantial central opaque region as padding removal. Keep
    # a full-image attempt as well: translucent margin content is not discarded.
    if bbox and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= size[0] * size[1] * 0.5:
        return bbox
    return full


@contextmanager
def _ocr_source(path: Path, detection: dict | None, mode: str):
    with image_rgb(path) as source:
        if detection is not None and mode == "mask" and not detection.get("whole_image_fallback"):
            from realopen_lens.ocr.regions import mask_regions

            with mask_regions(source, detection) as masked:
                yield masked
        else:
            yield source


def extract(
    path: Path,
    languages: str = "eng+fra+ara",
    threads: int = 2,
    timeout: float = 30,
    *,
    detector: Any = None,
    region_mode: str = "gate",
) -> dict[str, Any]:
    if not 1 <= threads <= 32 or timeout <= 0:
        raise ValueError("OCR requires 1–32 threads and a positive timeout")
    if region_mode not in {"mask", "gate"}:
        raise ValueError("Text region mode must be 'mask' or 'gate'")
    executable = runtime.resolve(shutil.which("tesseract"))
    if executable is None:
        raise ValueError("Tesseract 5 is required for OCR")
    environment = _environment(executable, languages)
    start = monotonic()
    attempts: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    recognition_attempts = 0
    warnings = []
    detection = None
    with ExitStack() as buffers:

        def keep(image: Image.Image) -> Image.Image:
            buffers.callback(image.close)
            return image

        with image_rgb(path) as source:
            original_size = source.size
            if detector is not None:
                detection = detector.detect(source)
                if detection["candidate_limit_reached"]:
                    warnings.append("Detector candidate limit reached; preserving full image")
                elif not detection["regions"] and max(
                    detection.get("input_size", original_size)
                ) < max(original_size):
                    # A reduced detector input can erase small glyphs that Tesseract
                    # still reads from the source. No regions is insufficient evidence
                    # to discard recognition (or mask away the source) in this case.
                    detection = {
                        **detection,
                        "whole_image_fallback": "no_regions_after_downscaling",
                    }
                    warnings.append(
                        "No text regions after detector downscaling; preserving whole-image OCR"
                    )
                elif not detection["regions"]:
                    with Image.open(path) as original:
                        if getattr(original, "n_frames", 1) > 1:
                            warnings.append("Only first frame/page processed")
                    return {
                        **environment,
                        "text": "",
                        "words": [],
                        "confidence": None,
                        "warnings": warnings
                        + ["No text regions detected; text absence is not established"],
                        "original_size": list(original_size),
                        "source_region": [0, 0, *original_size],
                        "ocr_size": list(original_size),
                        "rotation": 0,
                        "selected_psm": None,
                        "inference_skipped": "no_detected_text_regions",
                        "text_detection": detection,
                        "region_mode": region_mode,
                        "alternate_texts": [],
                        "attempts": [],
                        "failed_attempts": [],
                    }
                if region_mode == "mask" and not detection.get("whole_image_fallback"):
                    from realopen_lens.ocr.regions import mask_regions

                    source = keep(mask_regions(source, detection))
            region = _source_region(path, original_size)
            full_region = (0, 0, *original_size)
            rgb = keep(source.copy())
            rgb.thumbnail((2400, 2400))
            working = keep(source.crop(region)) if region != full_region else rgb
            working.thumbnail((2400, 2400))
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) > 1:
                warnings.append("Only first frame/page processed")

        def attempt(
            image: Image.Image,
            psm: int,
            label: str,
            source_region: tuple,
            attempt_timeout: float = 15,
            **metadata: Any,
        ) -> dict | None:
            nonlocal recognition_attempts
            remaining = timeout - (monotonic() - start)
            if remaining <= 0 or recognition_attempts >= 4:
                warnings.append("OCR fallback budget exhausted")
                return None
            try:
                recognition_attempts += 1
                result = _recognize(
                    image, executable, languages, threads, psm, min(15, remaining, attempt_timeout)
                )
                result.update(input=label, source_region=list(source_region), **metadata)
                attempts.append(result)
                return result
            except (subprocess.TimeoutExpired, ValueError) as error:
                if label == "rgb" and not isinstance(error, subprocess.TimeoutExpired):
                    raise
                warnings.append(f"OCR fallback failed: {type(error).__name__}")
                failed_attempts.append(
                    {
                        "psm": psm,
                        "input": label,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                )
                return None

        first = attempt(rgb, 3, "rgb", full_region)
        initial_recovered = first is None
        if first is None:
            # A second full-page pass can repeat the same timeout. Split the remaining
            # budget across disjoint bands, retaining explicitly partial evidence.
            initial_gray, _ = _gray(rgb)
            keep(initial_gray)
            recovered = []
            for number in range(3):
                remaining = timeout - (monotonic() - start)
                if remaining <= 0:
                    break
                top, bottom = number * rgb.height // 3, (number + 1) * rgb.height // 3
                if bottom <= top:
                    continue
                with initial_gray.crop((0, top, rgb.width, bottom)) as band:
                    candidate = attempt(
                        band,
                        11,
                        f"timeout-band-{number}",
                        (
                            0,
                            round(top * original_size[1] / rgb.height),
                            original_size[0],
                            round(bottom * original_size[1] / rgb.height),
                        ),
                        attempt_timeout=remaining / (3 - number),
                    )
                if candidate is not None and candidate["words"]:
                    recovered.append((number, top, candidate))
            if not recovered:
                raise subprocess.TimeoutExpired("OCR timeout recovery exhausted", timeout)
            words = [
                {**word, "box": [word["box"][0], word["box"][1] + top, *word["box"][2:]]}
                for _, top, candidate in recovered
                for word in candidate["words"]
            ]
            first = {
                "text": "\n".join(candidate["text"] for _, _, candidate in recovered),
                "words": words,
                "confidence": sum(w["confidence"] for w in words) / len(words),
                "psm": 11,
                "ocr_size": list(rgb.size),
                "source_region": list(full_region),
                "input": "timeout-band-recovery",
                "aggregate": True,
                "seconds": sum(c["seconds"] for _, _, c in recovered),
                "timeout_recovery": {"bands": 3, "recovered_bands": [i for i, _, _ in recovered]},
            }
            attempts.append(first)
            warnings.append(
                "Whole-image OCR timed out; recovered partial band text, not a complete transcript"
            )
        work_result = first
        cropped = region != full_region
        if cropped and not initial_recovered:
            work_result = attempt(working, 3, "opaque-region-rgb", region) or first
        gray, inverted = _gray(working)
        keep(gray)
        incomplete = _layout_incomplete(gray, work_result["words"])
        if not initial_recovered and (
            cropped
            or inverted
            or incomplete
            or len(work_result["words"]) < 3
            or (work_result["confidence"] or 0) < 55
        ):
            cleaned, removed = (
                _remove_rules(gray) if incomplete and not inverted else (gray.copy(), False)
            )
            keep(cleaned)
            attempt(cleaned, 11, "grayscale-sparse", region, rules_removed=removed)

        best = max(attempts, key=_quality)
        if (
            not initial_recovered
            and max(region[2] - region[0], region[3] - region[1]) > 3000
            and len(best["words"]) >= 8
        ):
            with _ocr_source(path, detection, region_mode) as source:
                large = keep(source.crop(region))
            large.thumbnail((3200, 3200))
            large_gray, _ = _gray(large)
            keep(large_gray)
            attempt(large_gray, 11, "high-resolution-grayscale", region)

        orientation = None
        best = max(attempts, key=_quality)
        remaining = timeout - (monotonic() - start)
        if (
            not initial_recovered
            and len(best["words"]) >= 8
            and (best["confidence"] or 0) < 55
            and remaining > 2
            and len(attempts) < 4
        ):
            try:
                with _ocr_source(path, detection, region_mode) as source:
                    orientation_source = keep(source.crop(tuple(best["source_region"])))
                orientation_source.thumbnail(tuple(best.get("ocr_size", gray.size)))
                orientation = _orientation(
                    orientation_source, executable, threads, min(3, remaining)
                )
                if orientation["rotation"] in (90, 180, 270) and orientation["confidence"] >= 2:
                    orientation_gray, _ = _gray(orientation_source)
                    keep(orientation_gray)
                    rotated = keep(
                        orientation_gray.rotate(
                            -orientation["rotation"], expand=True, fillcolor=255
                        )
                    )
                    candidate = attempt(
                        rotated,
                        3,
                        "orientation",
                        tuple(best["source_region"]),
                        rotation=orientation["rotation"],
                    )
                    if candidate is not None and _quality(candidate) <= _quality(best) * 1.25:
                        attempts.remove(candidate)
                        failed_attempts.append(
                            {
                                "stage": "orientation",
                                "rotation": orientation["rotation"],
                                "reason": "No substantial gain in readable evidence",
                                "text": candidate["text"],
                            }
                        )
            except (subprocess.TimeoutExpired, ValueError) as error:
                failed_attempts.append(
                    {
                        "stage": "orientation",
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                )
        selected = first if initial_recovered else max(attempts, key=_quality)
        if not selected["words"]:
            warnings.append("Empty OCR output; text absence has not been established")
        elif (selected["confidence"] or 0) < 55:
            warnings.append("Low OCR confidence; extracted text may be unusable")
        # Preserve observed, confidence-filtered alternatives for code/keyword
        # lookup. They are not presented or scored as the canonical transcript.
        alternatives = list(
            dict.fromkeys(
                " ".join(word["text"] for word in candidate["words"] if word["confidence"] >= 70)
                for candidate in attempts
                if candidate is not selected and not initial_recovered
            )
        )
        return {
            **environment,
            **({"timeout_recovery": selected["timeout_recovery"]} if initial_recovered else {}),
            **({"text_detection": detection, "region_mode": region_mode} if detection else {}),
            "text": selected["text"],
            "words": selected["words"],
            "confidence": selected["confidence"],
            "warnings": warnings,
            "original_size": list(original_size),
            "ocr_size": selected.get("ocr_size", list(gray.size)),
            "source_region": selected["source_region"],
            "selected_psm": selected["psm"],
            "rotation": selected.get("rotation", 0),
            "orientation": orientation,
            "preprocessing": {
                "alpha_background": "white",
                "rgb_primary": True,
                "grayscale_fallback": True,
                "inverted": inverted,
                "layout_fallback_requested": incomplete,
                "opaque_region": list(region),
            },
            "alternate_texts": [text for text in alternatives if text],
            "attempts": [
                {key: value for key, value in candidate.items() if key != "words"}
                for candidate in attempts
            ],
            "failed_attempts": failed_attempts,
        }
