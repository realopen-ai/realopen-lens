"""Local Tesseract OCR with bounded retries and explicit low-quality diagnostics."""

import math


import re


import subprocess


from functools import lru_cache


from pathlib import Path


from typing import Any


from realopen_lens.models import digest


@lru_cache(maxsize=8)
def _environment(executable: str, languages: str) -> dict[str, Any]:
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=True, timeout=10
    )
    result = subprocess.run(
        [executable, "--list-langs"], capture_output=True, text=True, check=True, timeout=10
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
    return {"version": version.stdout.splitlines()[0], "language_pack_sha256": hashes}


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
