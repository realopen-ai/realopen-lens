"""Source fingerprints, dependency versions, and experimental engine names."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def source_hash(root: Path) -> str:
    value = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


ENGINES = ("tesseract", "rapidocr", "glm-ocr")
