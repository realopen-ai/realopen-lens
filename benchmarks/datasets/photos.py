"""Explicit, reproducible public-image downloads for provisional retrieval tests."""

import argparse
import hashlib
import io
import json
import os
import random
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from PIL import Image

ANNOTATIONS_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
ANNOTATIONS_SHA256 = "4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817"
IMAGE_BASE = "https://s3.amazonaws.com/images.cocodataset.org/val2014/"
USER_AGENT = "RealOpen-Lens-Benchmark/0.1 (public COCO sample download)"


def get_bytes(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"Download exceeded {limit} bytes")
    return content


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def annotations(cache: Path) -> list[dict[str, Any]]:
    archive = cache / "caption_datasets.zip"
    if archive.is_file():
        payload = archive.read_bytes()
        if sha256(payload) != ANNOTATIONS_SHA256:
            raise ValueError(f"Annotation cache checksum mismatch: {archive}")
    else:
        payload = get_bytes(ANNOTATIONS_URL, 64 * 1024 * 1024)
        if sha256(payload) != ANNOTATIONS_SHA256:
            raise ValueError("Downloaded annotations do not match the pinned checksum")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        if bundle.getinfo("dataset_coco.json").file_size > 256 * 1024 * 1024:
            raise ValueError("Annotation JSON is unexpectedly large")
        return json.loads(bundle.read("dataset_coco.json"))["images"]


def select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    candidates = sorted(
        (
            row
            for row in rows
            if row.get("split") == "test"
            and row.get("filepath") == "val2014"
            and re.fullmatch(r"COCO_val2014_\d{12}\.jpg", row.get("filename", ""))
            and row.get("sentences")
        ),
        key=lambda row: row["filename"],
    )
    if not 1 <= count <= min(500, len(candidates)):
        raise ValueError(f"Count must be between 1 and {min(500, len(candidates))}")
    return random.Random(seed).sample(candidates, count)


def download_image(row: dict[str, Any], output: Path) -> dict[str, Any]:
    name = row["filename"]
    target = output / name
    if target.exists():
        raise ValueError(f"Refusing to overwrite an unverified existing image: {target}")
    url = IMAGE_BASE + name
    payload = get_bytes(url, 10 * 1024 * 1024)
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
        if image.format != "JPEG" or width * height > 24_000_000:
            raise ValueError(f"Unexpected image format or dimensions: {name}")
        image.verify()
    temporary = target.with_suffix(".jpg.part")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return {
        "filename": name,
        "coco_id": row["cocoid"],
        "url": url,
        "source_page": f"https://cocodataset.org/#explore?id={row['cocoid']}",
        "sha256": sha256(payload),
        "bytes": len(payload),
        "width": width,
        "height": height,
        "captions": [sentence["raw"].strip() for sentence in row["sentences"]],
    }


def save_json(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fetch(output: Path, cache: Path, count: int, seed: int, workers: int) -> Path:
    if not 1 <= count <= 500:
        raise ValueError("Count must be between 1 and 500")
    selected = select(annotations(cache), count, seed)
    manifest_path = output / "download-manifest.json"
    identity = {
        "source": "COCO 2014 / Karpathy test split",
        "seed": seed,
        "count": count,
        "annotations_url": ANNOTATIONS_URL,
        "annotations_sha256": ANNOTATIONS_SHA256,
    }
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if previous and any(previous.get(key) != value for key, value in identity.items()):
        raise ValueError("Output belongs to a different sample. Choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    images = {}
    selected_names = {row["filename"] for row in selected}
    for item in previous.get("images", []):
        if item["filename"] not in selected_names:
            raise ValueError("Manifest contains an image outside the selected sample")
        path = output / item["filename"]
        if path.is_file():
            if sha256(path.read_bytes()) != item["sha256"]:
                raise ValueError(f"Existing image checksum mismatch: {path}")
            images[item["filename"]] = item
    failures = {}
    pending = [row for row in selected if row["filename"] not in images]
    print(f"Sample: {count} images; seed {seed}; {len(images)} verified cached images", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(download_image, row, output): row for row in pending}
        for task in as_completed(tasks):
            row = tasks[task]
            name = row["filename"]
            try:
                images[name] = task.result()
                print(f"Downloaded {name} ({len(images)}/{count})", flush=True)
            except (OSError, ValueError) as error:
                failures[name] = str(error)
                print(f"Failed {name}: {error}", flush=True)
            save_json(
                manifest_path,
                {
                    **identity,
                    "images": sorted(images.values(), key=lambda item: item["filename"]),
                    "failures": failures,
                    "source_terms": "https://cocodataset.org/#termsofuse",
                    "scope": (
                        "Photographs only; captions supply known positives, "
                        "not exhaustive judgments."
                    ),
                },
            )
    if failures:
        raise ValueError(
            f"{len(failures)} downloads failed. Rerun the same command to retry missing files."
        )
    query_items = [
        {
            "id": f"coco-{item['coco_id']}",
            "text": item["captions"][0],
            "language": "en",
            "relevant": [item["filename"]],
            "judgment_source": "Published COCO caption; known positive only",
        }
        for item in sorted(images.values(), key=lambda item: item["filename"])
    ]
    query_file = output / "queries.caption-derived.json"
    if query_file.exists() and json.loads(query_file.read_text()) != query_items:
        raise ValueError(f"Refusing to overwrite edited query judgments: {query_file}")
    save_json(query_file, query_items)
    print(f"Ready: {output}; queries: {query_file}", flush=True)
    return query_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/public-sample"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/sample-downloads"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    try:
        fetch(args.output, args.cache, args.count, args.seed, args.workers)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
