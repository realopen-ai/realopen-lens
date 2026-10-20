"""Explicit benchmark downloads and content-integrity checks."""

import hashlib
import subprocess
import re
from pathlib import Path


class DownloadRateLimitError(RuntimeError):
    def __init__(self, url: str, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"HTTP 429 for {url}; retry after {retry_after} seconds")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(
    url: str,
    path: Path,
    blob: str | None = None,
    *,
    max_bytes: int = 350000000,
    retries: int = 2,
    headers: Path | None = None,
) -> Path:
    if max_bytes < 1 or not 0 <= retries <= 5:
        raise ValueError("Download size limit must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        try:
            subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "--retry",
                    str(retries),
                    "--max-time",
                    "90",
                    "--max-filesize",
                    str(max_bytes),
                    url,
                    "-o",
                    str(part),
                    *(["--dump-header", str(headers)] if headers is not None else []),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            if headers is not None and headers.exists():
                content = headers.read_text(encoding="utf-8", errors="replace")[:128000]
                if re.search(r"HTTP/\S+ 429", content):
                    delay = re.findall(r"(?im)^retry-after:\s*(\d+)\s*$", content)
                    raise DownloadRateLimitError(url, int(delay[-1]) if delay else 600) from error
            raise
        if blob:
            payload = part.read_bytes()
            actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if actual != blob:
                raise ValueError(f"Git object integrity mismatch: {url}")
        part.replace(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"Cached download exceeds its size limit: {path}")
    if blob:
        payload = path.read_bytes()
        if hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() != blob:
            raise ValueError(f"Cached Git object integrity mismatch: {path}")
    return path
