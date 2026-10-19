"""Image metadata, source identity, and family-separated split helpers."""

import hashlib
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageFont
from benchmarks.datasets.downloads import digest


def font(size: int, path: Path) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.BASIC)


def family_split(family: str) -> str:
    return "holdout" if int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 5 == 0 else "dev"


def metadata(
    path: Path,
    root: Path,
    category: str,
    subcategory: str,
    source: dict,
    family: str | None = None,
    **extra: Any,
) -> dict:
    relative = path.relative_to(root).as_posix()
    with Image.open(path) as image:
        width, height = image.size
        mode, fmt, frames = image.mode, image.format, getattr(image, "n_frames", 1)
        gray = image.convert("L").resize((9, 8))
        values = np.asarray(gray)
        dhash = sum(int(v) << i for i, v in enumerate((values[:, 1:] > values[:, :-1]).flat))
    family = family or relative
    return {
        "id": relative,
        "path": relative,
        "category": category,
        "subcategory": subcategory,
        "width": width,
        "height": height,
        "format": fmt,
        "mode": mode,
        "frames": frames,
        "sha256": digest(path),
        "dhash": f"{dhash:016x}",
        "source": source,
        "family": family,
        "split": family_split(family),
        "difficulty": [],
        "expected": {"decode": "reject" if width * height > 24_000_000 else "success"},
        **extra,
    }


def annotation_matches(row: dict, annotation: dict) -> bool:
    """A basename is not a source identity (e.g. unrelated header.png images)."""
    source = row["source"]
    return (
        source.get("repo") == annotation["repo"] and source.get("path") == annotation["source_path"]
    )
