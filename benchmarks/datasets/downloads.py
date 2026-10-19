"""Explicit benchmark downloads and content-integrity checks."""

import hashlib
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, path: Path, blob: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "2",
                "--max-time",
                "90",
                "--max-filesize",
                "350000000",
                url,
                "-o",
                str(part),
            ],
            check=True,
            capture_output=True,
        )
        if blob:
            payload = part.read_bytes()
            actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if actual != blob:
                raise ValueError(f"Git object integrity mismatch: {url}")
        part.replace(path)
    if blob:
        payload = path.read_bytes()
        if hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() != blob:
            raise ValueError(f"Cached Git object integrity mismatch: {path}")
    return path
