"""Populate an explicit build cache with size/hash-verified, pinned OCR data."""

import argparse
import json
from pathlib import Path
import tempfile

from bundle_ocr import checksum, language_data


def prepare(cache: Path):
    registry = json.loads(Path(__file__).with_name("tessdata.json").read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    if all(
        (path := cache / f"{name}.traineddata").is_file()
        and path.stat().st_size == spec["size"]
        and checksum(path) == spec["sha256"]
        for name, spec in registry.items()
    ):
        print("Pinned OCR language cache verified", flush=True)
        return
    # Stage on the same filesystem; never cache partially downloaded assets.
    with tempfile.TemporaryDirectory(prefix=".tessdata-", dir=cache) as directory:
        staged = Path(directory) / "verified"
        language_data(staged, cache, registry)
        for name in registry:
            (staged / f"{name}.traineddata").replace(cache / f"{name}.traineddata")
    print("Pinned OCR language cache populated and verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    prepare(parser.parse_args().cache_dir)
