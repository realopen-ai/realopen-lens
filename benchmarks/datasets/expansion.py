"""Restore or build an explicitly curated expansion without changing the legacy corpus."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from benchmarks.datasets.downloads import digest, download, DownloadRateLimitError
from benchmarks.datasets.metadata import metadata
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.scale import pixel_hash
from realopen_lens.encoders.onnx import image_rgb


def safe_path(root: Path, relative: str | Path) -> Path:
    path = root / relative
    if path.resolve() == root.resolve() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset paths must stay inside their root")
    return path


def preview(path: Path) -> tuple[int, float, np.ndarray]:
    with image_rgb(path) as image:
        if image.width * image.height > 24_000_000:
            raise ValueError("Expansion image exceeds the production pixel limit")
        gray = image.convert("L")
        values = np.asarray(gray.resize((9, 8)))
        dhash = sum(int(v) << i for i, v in enumerate((values[:, 1:] > values[:, :-1]).flat))
        return dhash, image.width / image.height, np.asarray(image.resize((64, 64)), dtype=np.int16)


def near_duplicate(a: tuple, b: tuple) -> bool:
    """Confirm a cheap dHash match with aspect ratio and RGB pixel agreement.

    dHash alone over-groups unrelated text pages. This guard catches close visual
    copies, while explicit work/project identities group translations and templates.
    It is not an exhaustive duplicate detector.
    """
    return bool(
        (a[0] ^ b[0]).bit_count() <= 4
        and abs(a[1] / b[1] - 1) <= 0.05
        and np.abs(a[2] - b[2]).mean() <= 3
    )


def copy_verified(source: Path, target: Path, sha256: str) -> None:
    if digest(source) != sha256:
        raise ValueError(f"Legacy source changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        # Hard links change legacy ctime and invalidate its saved source signatures.
        shutil.copy2(source, target)
    if digest(target) != sha256:
        raise ValueError(f"Existing expansion image differs: {target}")


def build(
    base: Path,
    base_manifest: Path,
    sources: Path,
    output: Path,
    workers: int = 1,
    download_delay: float = 10,
) -> dict:
    if not 1 <= workers <= 4 or not np.isfinite(download_delay) or not 0 <= download_delay <= 60:
        raise ValueError("Use one to four download workers and a delay from zero to 60 seconds")
    legacy = json.loads(base_manifest.read_text(encoding="utf-8"))
    catalog = json.loads(sources.read_text(encoding="utf-8"))
    if catalog["base_corpus_sha256"] != digest(base_manifest):
        raise ValueError("Source catalog identifies a different legacy corpus")
    output.mkdir(parents=True, exist_ok=True)
    pause = output / "download-pause.json"
    if pause.exists():
        remaining = json.loads(pause.read_text(encoding="utf-8"))["not_before"] - time.time()
        uncached = any(
            not safe_path(
                output,
                Path(item["category"])
                / item["source"].get("provider_tag", "commons")
                / item["group"]
                / item["filename"],
            ).exists()
            for item in catalog["images"]
        )
        if remaining > 0 and uncached:
            raise ValueError(f"Download server requested a pause; retry in {int(remaining) + 1}s")
    rows = deepcopy(legacy["images"])
    source_folds: dict[str, str] = {}
    for item in catalog["images"]:
        group, fold = item["group"], item["split"]
        if fold not in {"train", "validation", "test"} or (
            group in source_folds and source_folds[group] != fold
        ):
            raise ValueError("Source group crosses corpus folds")
        source_folds[group] = fold
    pixels = {row["pixel_sha256"] for row in rows}
    hashes = {row["sha256"] for row in rows}
    previous = []
    for row in rows:
        path = safe_path(output, row["path"])
        copy_verified(safe_path(base, row["path"]), path, row["sha256"])
        previous.append((row, preview(path)))
    stopped = threading.Event()
    error_lock = threading.Lock()
    errors = []

    def acquire(item):
        provider = item["source"].get("provider_tag", "commons")
        relative = Path(item["category"]) / provider / item["group"] / item["filename"]
        path = safe_path(output, relative)
        if not path.exists():
            time.sleep(download_delay)
        # Stop the batch on transfer failures; do not hammer a rate-limited server.
        download(
            item["source"]["url"],
            path,
            max_bytes=32 * 1024 * 1024,
            retries=0,
            headers=path.with_suffix(path.suffix + ".headers"),
        )
        if item["source"].get("original_sha1") and item["source"]["transform"] == "original":
            with path.open("rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha1").hexdigest()
                    != item["source"]["original_sha1"]
                ):
                    raise ValueError(f"Commons source version changed: {item['title']}")
        if item.get("sha256") and digest(path) != item["sha256"]:
            raise ValueError(f"Frozen Commons image changed: {item['title']}")
        candidate = preview(path)
        row = metadata(path, output, item["category"], "commons", item["source"])
        row.update(
            id=relative.as_posix(),
            split=item["split"],
            source_document=provider + ":" + item["group"],
            duplicate_group=hashlib.sha256(("retrieval-v2:" + item["group"]).encode()).hexdigest(),
            pixel_sha256=pixel_hash(path),
            content_language=item["content_language"],
            exposure="new-text-sources",
        )
        if min(row["width"], row["height"]) < 400:
            row["difficulty"].append("low-resolution")
        return row, candidate

    def guarded_acquire(item):
        if stopped.is_set():
            return None
        try:
            result = acquire(item)
            print(f"Acquired {item['title']}", flush=True)
            return result
        except Exception as error:
            stopped.set()
            evidence = getattr(error, "stderr", b"")
            if isinstance(evidence, bytes):
                evidence = evidence.decode("utf-8", errors="replace")
            with error_lock:
                if isinstance(error, DownloadRateLimitError):
                    until = time.time() + error.retry_after
                    if pause.exists():
                        until = max(
                            until, json.loads(pause.read_text(encoding="utf-8"))["not_before"]
                        )
                    save_json(
                        pause,
                        {
                            "not_before": until,
                            "retry_after_seconds": error.retry_after,
                            "error": str(error),
                        },
                    )
                errors.append(
                    {
                        "title": item["title"],
                        "error": str(error)[:1000],
                        "transfer_details": evidence[-1000:],
                    }
                )
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        additions = [row for row in pool.map(guarded_acquire, catalog["images"]) if row is not None]
    if errors or len(additions) != len(catalog["images"]):
        save_json(
            output / "acquisition-errors.json",
            {"errors": errors, "complete": len(additions), "expected": len(catalog["images"])},
        )
        raise ValueError("Acquisition incomplete; inspect acquisition-errors.json before resuming")
    groups = {row["duplicate_group"]: row["duplicate_group"] for row, _ in additions}

    def find(key):
        while groups[key] != key:
            groups[key] = groups[groups[key]]
            key = groups[key]
        return key

    for index, (row, image) in enumerate(additions):
        if row["sha256"] in hashes or row["pixel_sha256"] in pixels:
            raise ValueError(f"Duplicate expansion image: {row['path']}")
        if any(near_duplicate(image, other) for _, other in previous):
            raise ValueError(f"Expansion image is a close legacy copy: {row['path']}")
        for other, thumbnail in additions[:index]:
            if near_duplicate(image, thumbnail):
                if row["split"] != other["split"]:
                    raise ValueError(
                        f"Near duplicates cross splits: {row['path']}, {other['path']}"
                    )
                a, b = find(row["duplicate_group"]), find(other["duplicate_group"])
                groups[max(a, b)] = min(a, b)
        hashes.add(row["sha256"])
        pixels.add(row["pixel_sha256"])
    for row, _ in additions:
        row["duplicate_group"] = find(row["duplicate_group"])
        rows.append(row)
    manifest = {
        "version": 2,
        "seed": 20261020,
        "base_corpus_sha256": digest(base_manifest),
        "sources_sha256": digest(sources),
        "protocol_sha256": catalog["protocol_sha256"],
        "images": rows,
        "duplicate_policy": "Exact encoded and production-normalized pixels rejected; explicit "
        "source work/software groups plus dHash<=4, aspect within 5%, RGB64 MAE<=3 guard",
    }
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"images": len(rows), "new_sources": len(additions)}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/retrieval-2000-v1"))
    parser.add_argument(
        "--base-manifest", type=Path, default=Path("benchmarks/sources/retrieval-v1/corpus.json")
    )
    parser.add_argument(
        "--sources", type=Path, default=Path("benchmarks/sources/retrieval-v2/sources.json")
    )
    parser.add_argument("--output", type=Path, default=Path("data/retrieval-v2"))
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--download-delay", type=float, default=10)
    args = parser.parse_args()
    build(
        args.base, args.base_manifest, args.sources, args.output, args.workers, args.download_delay
    )


if __name__ == "__main__":
    main()
