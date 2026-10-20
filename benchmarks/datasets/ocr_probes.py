"""Freeze a small stratified OCR regression set plus explicit diagnostic images."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json


def prepare(dataset: Path, output: Path, extras: list[Path], seed: int = 20261020) -> dict:
    if output.exists():
        raise ValueError("Preserve frozen OCR inputs; choose a new output file")
    manifest = dataset / "manifest.json"
    corpus = json.loads(manifest.read_text(encoding="utf-8"))
    groups: dict[str, list] = defaultdict(list)
    for row in corpus["images"]:
        groups[row["category"]].append(row)
    rng, images = random.Random(seed), []
    for category, rows in sorted(groups.items()):
        for row in rng.sample(sorted(rows, key=lambda r: r["id"]), min(4, len(rows))):
            path = dataset / row["path"]
            if digest(path) != row["sha256"]:
                raise ValueError("OCR regression source differs from its manifest")
            images.append({**row, "path": str(path.resolve())})
    for path in extras:
        images.append(
            {
                "id": "diagnostic/" + path.name,
                "path": str(path.resolve()),
                "sha256": digest(path),
                "category": "explicit-diagnostic",
                "reference_kind": "no transcript supplied",
            }
        )
    config = {
        "version": 1,
        "seed": seed,
        "manifest_sha256": digest(manifest),
        "images": images,
        "selection": "Four fixed-seed images per category plus explicit diagnostics; "
        "no output-based substitution",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-image", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(len(prepare(args.dataset, args.output, args.extra_image)["images"]))


if __name__ == "__main__":
    main()
