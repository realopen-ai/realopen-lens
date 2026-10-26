"""Bounded offline script-group routing; confidence is not a language probability."""

import io
import math
from pathlib import Path
import re
import subprocess
from time import monotonic

import numpy as np
from PIL import Image, ImageOps

from realopen_lens.models.manager import digest
from realopen_lens.ocr import runtime
from realopen_lens.ocr.languages import LANGUAGES

GROUPS = {
    "Latin": ("eng", "fra", "spa", "por", "deu", "ita", "ind", "tur", "vie"),
    "Cyrillic": ("rus", "eng"),
    "Arabic": ("ara", "urd", "fas", "eng"),
    "Devanagari": ("hin", "eng"),
    "Bengali": ("ben", "eng"),
    "Han": ("chi_sim", "chi_tra", "jpn", "kor", "eng"),
    "Japanese": ("jpn", "chi_sim", "chi_tra", "eng"),
    "Korean": ("kor", "eng"),
}
ALIASES = {"Hiragana": "Japanese", "Katakana": "Japanese", "Hangul": "Korean"}
# A conservative margin above Tesseract's internal acceptance score (1.0).
# Sparse text can produce confident-looking wrong groups; manual mode is default.
MIN_CONFIDENCE = 3.0


def policy() -> dict:
    path = Path(__file__)
    source = path.with_suffix(".py") if path.suffix in {".pyc", ".pyo"} else path
    return {
        "version": 2,
        "source_sha256": digest(source if source.is_file() else path),
        "groups": {name: list(codes) for name, codes in GROUPS.items()},
        "confidence_threshold": MIN_CONFIDENCE,
        "max_probes": 3,
        "max_seconds": 3,
    }


def available(installed: list[str]) -> list[str]:
    return [code for code in LANGUAGES if code in installed]


def choose(probes: list[dict], installed: list[str]) -> dict:
    supported = available(installed)
    if not supported:
        raise ValueError("Automatic OCR needs at least one supported language pack")
    strong = [
        p for p in probes if p.get("status") == "ok" and p.get("confidence", 0) >= MIN_CONFIDENCE
    ]
    full = next((p for p in strong if p["region"] == "full"), None)
    groups = list(dict.fromkeys(ALIASES.get(p["script"], p["script"]) for p in strong))
    if full is None and not (len(strong) >= 2 and len(groups) == 1):
        return {
            "groups": [],
            "selected_languages": "+".join(supported),
            "fallback": "uncertain_script",
            "probes": probes,
        }
    if not groups or any(group not in GROUPS for group in groups):
        return {
            "groups": groups,
            "selected_languages": "+".join(supported),
            "fallback": "unsupported_script",
            "probes": probes,
        }
    if any(
        not (set(GROUPS[group]) - ({"eng"} if group != "Latin" else set())) & set(supported)
        for group in groups
    ):
        return {
            "groups": groups,
            "selected_languages": "+".join(supported),
            "fallback": "missing_group_packs",
            "probes": probes,
        }
    selected = list(
        dict.fromkeys(code for group in groups for code in GROUPS[group] if code in supported)
    )
    return {
        "groups": groups,
        "selected_languages": "+".join(selected),
        "fallback": None,
        "probes": probes,
    }


def detect(
    image: Image.Image, executable: str, installed: list[str], threads: int, timeout: float
) -> dict:
    start = monotonic()
    probes = []
    with ImageOps.grayscale(image) as gray:
        gray.thumbnail((1600, 1600))
        if float(np.median(np.asarray(gray))) < 128:
            inverted = ImageOps.invert(gray)
            gray.paste(inverted)
            inverted.close()
        regions = [("full", (0, 0, *gray.size))]
        if min(gray.size) >= 256:
            regions += [
                ("top", (0, 0, gray.width, gray.height // 2)),
                ("bottom", (0, gray.height // 2, gray.width, gray.height)),
            ]
        for name, box in regions:
            remaining = timeout - (monotonic() - start)
            if remaining <= 0:
                break
            payload = io.BytesIO()
            with gray.crop(box) as sample:
                sample.save(payload, format="PNG")
            try:
                result = subprocess.run(
                    [
                        executable,
                        "stdin",
                        "stdout",
                        "-l",
                        "osd",
                        "--psm",
                        "0",
                        "-c",
                        "min_characters_to_try=20",
                    ],
                    input=payload.getvalue(),
                    capture_output=True,
                    timeout=min(1.5, remaining),
                    env={**runtime.environment(executable), "OMP_THREAD_LIMIT": str(threads)},
                )
                output = result.stdout.decode("utf-8", errors="replace")
                script = re.search(r"^Script:\s*(\S+)", output, re.MULTILINE)
                confidence = re.search(r"^Script confidence:\s*([\d.]+)", output, re.MULTILINE)
                value = float(confidence[1]) if confidence else 0.0
                valid = result.returncode == 0 and script is not None and math.isfinite(value)
                probes.append(
                    {
                        "region": name,
                        "status": "ok" if valid else "unavailable",
                        "script": script[1] if script else "Unknown",
                        "confidence": value if math.isfinite(value) else 0,
                        "returncode": result.returncode,
                        "stderr": result.stderr.decode("utf-8", errors="replace")[:500],
                    }
                )
            except subprocess.TimeoutExpired:
                probes.append(
                    {"region": name, "status": "timeout", "script": "Unknown", "confidence": 0}
                )
    return {**choose(probes, installed), "seconds": monotonic() - start}
