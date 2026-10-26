"""Workspace OCR defaults, separate from each published folder configuration."""

import json
from pathlib import Path
import re

from realopen_lens.storage.artifacts import atomic_bytes


def validate(languages: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", languages) or len(languages) > 128:
        raise ValueError("Choose OCR language codes or automatic mode")
    if "auto" in languages.split("+") and languages != "auto":
        raise ValueError("Automatic mode cannot be combined with language codes")
    return languages


def read(root: Path, fallback: str = "") -> str:
    path = root / "ocr-preferences.json"
    if not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate(data["languages"])


def save(root: Path, languages: str) -> dict:
    data = {"version": 1, "languages": validate(languages)}
    atomic_bytes(root / "ocr-preferences.json", (json.dumps(data) + "\n").encode("utf-8"))
    return data
