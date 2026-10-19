"""Local image collection discovery."""

from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def scan(root: Path, limit: int) -> list[str]:
    if limit < 1:
        raise ValueError("Image scan limit must be positive")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    paths = sorted(
        str(p.relative_to(root).as_posix())
        for p in root.rglob("*")
        if p.suffix.lower() in EXTENSIONS and p.is_file() and p.resolve().is_relative_to(root)
    )
    if not paths:
        raise ValueError(f"No supported images in {root}. Add images to data/ and try again.")
    return paths[:limit]
