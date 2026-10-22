"""Explicit downloads and integrity checks; inference never calls the network."""

import hashlib
import json
import os
import urllib.request
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any


def registry() -> dict[str, Any]:
    return json.loads(files("realopen_lens.models").joinpath("models.lock.json").read_text())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifacts(
    precision: str, siglip_precision: str | None = None
) -> list[tuple[str, str, dict[str, Any]]]:
    if precision not in {"int8", "fp32"}:
        raise ValueError("Precision must be int8 or fp32")
    if siglip_precision is not None and siglip_precision not in {"int8", "fp32"}:
        raise ValueError("SigLIP2 precision must be int8 or fp32")
    return [
        (name, key, model["files"][key])
        for name, model in registry()["models"].items()
        for key in model["files"]
        if key == "tokenizer"
        or key.endswith(
            f"_{siglip_precision if name == 'siglip2' and siglip_precision else precision}"
        )
    ]


def artifact_path(root: Path, name: str, key: str) -> Path:
    model = registry()["models"][name]
    return root / name / model["revision"] / model["files"][key]["path"]


def verify(root: Path, precision: str, siglip_precision: str | None = None) -> None:
    for name, key, spec in artifacts(precision, siglip_precision):
        path = artifact_path(root, name, key)
        if not path.is_file():
            raise ValueError(f"Missing model file: {path}. Run 'lens models fetch' first.")
        if path.stat().st_size != spec["size"] or digest(path) != spec["sha256"]:
            raise ValueError(f"Model integrity check failed: {path}")


def fetch(
    root: Path,
    precision: str,
    progress: Callable[[str], None] = print,
    *,
    siglip_precision: str | None = None,
    download_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Download only explicit pinned artifacts, checking SHA-256 before publication."""
    for name, key, spec in artifacts(precision, siglip_precision):
        path = artifact_path(root, name, key)
        if path.is_file() and path.stat().st_size == spec["size"]:
            if digest(path) == spec["sha256"]:
                progress(f"Verified {name}/{spec['path']}")
                continue
        model = registry()["models"][name]
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{spec['path']}"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        progress(f"Downloading {name}/{spec['path']} ({spec['size'] / 1e6:.1f} MB)")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "realopen-lens/0.1"})
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as out,
            ):
                received = 0
                while block := response.read(1024 * 1024):
                    out.write(block)
                    received += len(block)
                    if download_progress:
                        download_progress(f"{name}/{spec['path']}", received, spec["size"])
            if temporary.stat().st_size != spec["size"] or digest(temporary) != spec["sha256"]:
                raise ValueError(f"Downloaded artifact failed integrity check: {name}/{key}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def fetch_text_detector(root: Path) -> Path:
    """Explicitly fetch one pinned detector, with bounded size and atomic publication."""
    from realopen_lens.ocr.regions import detector_spec

    spec = detector_spec()
    path = root / "text-detector" / spec["name"]
    if path.is_file() and path.stat().st_size == spec["size"] and digest(path) == spec["sha256"]:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".onnx.part")
    try:
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "realopen-lens/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            remaining = spec["size"]
            while block := response.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(block)
                if remaining < 0:
                    raise ValueError("Text detector download exceeded pinned size")
                out.write(block)
        if temporary.stat().st_size != spec["size"] or digest(temporary) != spec["sha256"]:
            raise ValueError("Text detector download failed integrity check")
        os.replace(temporary, path)
        return path
    finally:
        temporary.unlink(missing_ok=True)
