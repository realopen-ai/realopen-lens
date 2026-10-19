"""Build a licensed scale corpus; inference and evaluation never download images."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil

from benchmarks.datasets.downloads import digest, download
from benchmarks.datasets.licenses import load as licenses
from benchmarks.datasets.metadata import metadata
from benchmarks.datasets.photos import annotations, save_json

SEED = 20261020


def pixel_hash(path: Path) -> str:
    """Identify the oriented, alpha-composited RGB pixels actually used by the engine."""
    from realopen_lens.encoders.onnx import image_rgb

    with image_rgb(path) as rgb:
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def group_corpus(rows: list[dict]) -> None:
    """Conservative connected components: source families and dHash distance <=4.

    dHash is a leakage guard, not a claim that every connected image is a duplicate.
    Synthetic template families and repository families are intentionally kept together.
    """
    parents = {row["sha256"]: row["sha256"] for row in rows}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    families: dict[str, str] = {}
    for i, row in enumerate(rows):
        family = row["source_document"]
        if family in families:
            union(row["sha256"], families[family])
        families[family] = row["sha256"]
        for other in rows[:i]:
            if (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count() <= 4:
                union(row["sha256"], other["sha256"])
    for row in rows:
        row["duplicate_group"] = find(row["sha256"])
        value = (
            int(hashlib.sha256((str(SEED) + row["duplicate_group"]).encode()).hexdigest()[:8], 16)
            % 10
        )
        row["split"] = "train" if value < 6 else "validation" if value < 8 else "test"


def build(base: Path, output: Path, cache: Path, count: int, workers: int, lock: Path | None):
    if not 500 <= count <= 5000:
        raise ValueError("Use 500–5000 unique images")
    output.mkdir(parents=True, exist_ok=True)
    if lock is not None:
        frozen = json.loads(lock.read_text())
        if len(frozen["images"]) != count:
            raise ValueError("Requested count differs from frozen corpus")
        for row in frozen["images"]:
            path = output / row["path"]
            if not path.is_file():
                if row["origin"] == "base":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(base / row["path"], path)
                else:
                    download(row["source"]["url"], path)
            if digest(path) != row["sha256"] or pixel_hash(path) != row["pixel_sha256"]:
                raise ValueError(f"Frozen corpus mismatch: {row['id']}")
        save_json(output / "manifest.json", frozen)
        print(f"Verified {count} frozen unique images", flush=True)
        return
    if (output / "manifest.json").exists():
        return build(base, output, cache, count, workers, output / "manifest.json")
    base_manifest = json.loads((base / "manifest.json").read_text())
    rows, seen, pixels = [], set(), set()
    for item in base_manifest["images"]:
        if item["expected"]["decode"] != "success" or item["sha256"] in seen:
            continue
        source = base / item["path"]
        if digest(source) != item["sha256"]:
            raise ValueError("Base corpus changed")
        pixel = pixel_hash(source)
        if pixel in pixels:
            continue
        path = output / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)
        row = {
            key: item[key]
            for key in (
                "id",
                "path",
                "category",
                "subcategory",
                "width",
                "height",
                "format",
                "sha256",
                "dhash",
                "source",
                "difficulty",
            )
        }
        row.update(origin="base", pixel_sha256=pixel, source_document=item["family"])
        rows.append(row)
        seen.add(row["sha256"])
        pixels.add(pixel)
    info = licenses(cache)
    allowed = {x["id"]: x for x in info["licenses"] if x["id"] in (4, 5, 7, 8)}
    sources = {x["id"]: x for x in info["images"] if x["license"] in allowed}
    candidates = sorted(
        (
            r
            for r in annotations(cache)
            if r.get("filepath") == "val2014" and r.get("cocoid") in sources
        ),
        key=lambda r: r["filename"],
    )
    random.Random(SEED).shuffle(candidates)

    def fetch(item):
        url = "https://s3.amazonaws.com/images.cocodataset.org/val2014/" + item["filename"]
        path = output / "photos" / item["filename"]
        download(url, path)
        record = sources[item["cocoid"]]
        row = metadata(
            path,
            output,
            "photos",
            "coco-scale",
            {
                "kind": "real",
                "url": url,
                "dataset": "COCO2014",
                "coco_id": item["cocoid"],
                "license": allowed[record["license"]],
                "flickr_url": record.get("flickr_url"),
                "license_annotations_sha256": digest(cache / "coco-license-info.json"),
            },
        )
        return {
            key: row[key]
            for key in (
                "id",
                "path",
                "category",
                "subcategory",
                "width",
                "height",
                "format",
                "sha256",
                "dhash",
                "source",
                "difficulty",
            )
        } | {
            "origin": "download",
            "source_document": row["id"],
            "pixel_sha256": pixel_hash(path),
        }

    # Batches bound outstanding work and preserve seeded order, even across retries.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        offset = 0
        while len(rows) < count:
            pending: list[dict] = []
            while len(pending) < min(workers * 4, count - len(rows)):
                if offset >= len(candidates):
                    raise ValueError("Insufficient unique licensed photos")
                item = candidates[offset]
                offset += 1
                if ("photos/" + item["filename"]) not in {r["id"] for r in rows}:
                    pending.append(item)
            for row in pool.map(fetch, pending):
                if row["sha256"] in seen or row["pixel_sha256"] in pixels:
                    (output / row["path"]).unlink()
                    continue
                rows.append(row)
                seen.add(row["sha256"])
                pixels.add(row["pixel_sha256"])
            print(f"Acquired {len(rows)}/{count} distinct images", flush=True)
    group_corpus(rows)
    manifest = {
        "version": 1,
        "seed": SEED,
        "base_manifest_sha256": digest(base / "manifest.json"),
        "uniqueness": (
            "Distinct encoded SHA256 and production-normalized RGB pixels; "
            "near duplicates grouped"
        ),
        "grouping": (
            "Source family + transitive dHash Hamming distance <=4, " "split 60/20/20 by component"
        ),
        "images": sorted(rows, key=lambda r: r["id"]),
        "categories": dict(Counter(r["category"] for r in rows)),
    }
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"images": len(rows), "categories": manifest["categories"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/benchmark-500-v2"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval-2000-v1"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/hardening"))
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    build(args.base, args.output, args.cache, args.count, args.workers, args.lock)


if __name__ == "__main__":
    main()
