"""Explicitly fetch checksum-pinned official models for optional reference evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from benchmarks.datasets.downloads import digest


def fetch(lock: Path, root: Path) -> None:
    models = json.loads(lock.read_text(encoding="utf-8"))["models"]
    for name, model in models.items():
        for filename, spec in model["files"].items():
            path = root / name / model["revision"] / filename
            if (
                path.is_file()
                and path.stat().st_size == spec["size"]
                and digest(path) == spec["sha256"]
            ):
                print("Verified", name, filename, flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{filename}"
            print(f"Downloading {name}/{filename} ({spec['size'] / 1e6:.1f} MB)", flush=True)
            checksum, remaining = hashlib.sha256(), spec["size"]
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "realopen-lens/0.1"})
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    temporary.open("wb") as stream,
                ):
                    while block := response.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(block)
                        if remaining < 0:
                            raise ValueError("Reference model exceeded its pinned size")
                        checksum.update(block)
                        stream.write(block)
                if remaining or checksum.hexdigest() != spec["sha256"]:
                    raise ValueError("Reference model failed its pinned checksum")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path, default=Path("benchmarks/sources/encoder-reference.lock.json")
    )
    parser.add_argument("--models", type=Path, default=Path("models/encoder-reference"))
    args = parser.parse_args()
    fetch(args.lock, args.models)


if __name__ == "__main__":
    main()
