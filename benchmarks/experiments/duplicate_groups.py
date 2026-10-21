"""Real-index duplicate diagnostic; isolated snapshot, individual results and invariants."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from importlib.metadata import version

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from time import perf_counter

from benchmarks.experiments.virtual_collections import copy_snapshot
from realopen_lens.organization import duplicates
from realopen_lens.retrieval.session import SearchSession
from realopen_lens.retrieval.images import search_similar
from realopen_lens.models.manager import digest
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog
from realopen_lens.storage.artifacts import atomic_bytes


def image_checks(root: Path, collection: str, models: Path, scan: dict) -> list[dict]:
    catalog = Catalog(root, readonly=True)
    try:
        _, active = catalog.snapshot(collection)
        manifest = load_manifest(catalog, active)
    finally:
        catalog.close()
    query_id = scan["exact"][0]["image_ids"][0] if scan["exact"] else next(iter(scan["pixels"]))
    query_pixels = scan["pixels"][query_id]
    aliases = {key for key, pixels in scan["pixels"].items() if pixels == query_pixels}
    records = []
    for include in (False, True):
        started = perf_counter()
        results = search_similar(
            image_id=query_id,
            index_dir=root,
            collection=collection,
            models=models,
            include_self=include,
            limit=50,
        )
        assert len([r for r in results if r.image_id in aliases]) == int(include)
        records.append(
            {
                "kind": "indexed",
                "include_self": include,
                "seconds": perf_counter() - started,
                "status": "passed",
                "actual": [asdict(r) for r in results],
            }
        )
    row = next(r for r in manifest["images"] if r["image_id"] == query_id)
    query = root.parent / "external-reencoded.png"
    with Image.open(Path(manifest["config"]["root"]) / row["paths"][0]) as source:
        info = PngInfo()
        info.add_text("comment", "duplicate validation: new encoded bytes")
        source.convert("RGBA").save(query, pnginfo=info)
    started = perf_counter()
    results = search_similar(query, index_dir=root, collection=collection, models=models, limit=50)
    assert not any(r.image_id in aliases for r in results)
    records.append(
        {
            "kind": "external_reencoded",
            "seconds": perf_counter() - started,
            "status": "passed",
            "actual": [asdict(r) for r in results],
        }
    )
    return records


def run(source: Path, output: Path, collection: str, models: Path):
    metadata = copy_snapshot(source, output / "index", collection)
    records = []
    queries = ["pizza", "invoice", "terminal error", "فاتورة", "paysage"]
    with SearchSession(index_dir=output / "index", collection=collection, models=models) as session:
        baseline = {}
        for query in queries:
            baseline[query] = [asdict(r) for r in session.search(query, limit=50)]
        started = perf_counter()
        scan = duplicates.scan(output / "index", collection)
        scan_seconds = perf_counter() - started
        for query in queries:
            started = perf_counter()
            result = [asdict(r) for r in session.search(query, limit=50)]
            expected_ids = [r["image_id"] for r in baseline[query]]
            # Representatives must keep their relative baseline order; backfill can
            # introduce later candidates. No qrels/accuracy claim is made here.
            prefix = [r["image_id"] for r in result if r["image_id"] in expected_ids]
            assert prefix == sorted(prefix, key=expected_ids.index)
            all_ids = [key for r in result for key in [r["image_id"], *r["duplicate_ids"]]]
            assert len(all_ids) == len(set(all_ids))
            records.append(
                {
                    "query": query,
                    "baseline": baseline[query],
                    "actual": result,
                    "seconds": perf_counter() - started,
                    "status": "passed",
                }
            )
    atomic_bytes(
        output / "queries.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode(),
    )
    image_records = image_checks(output / "index", collection, models, scan)
    atomic_bytes(
        output / "image-queries.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in image_records).encode(),
    )
    summary = {
        **metadata,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("Pillow", "numpy", "onnxruntime")},
        "source_sha256": {
            str(path): digest(path)
            for path in (
                Path("src/realopen_lens/organization/duplicates.py"),
                Path("src/realopen_lens/organization/fingerprints.py"),
                Path("src/realopen_lens/retrieval/search.py"),
                Path("src/realopen_lens/retrieval/images.py"),
            )
        },
        "scan_seconds": scan_seconds,
        "fingerprinted": scan["fingerprinted"],
        "errors": scan["errors"],
        "unsupported": scan["unsupported"],
        "exact_groups": len(scan["exact"]),
        "collapsed_identities": sum(len(g["image_ids"]) - 1 for g in scan["exact"]),
        "near_pairs": len(scan["near"]),
        "queries": len(records),
        "image_queries": len(image_records),
        "purpose": "functional diagnostics; suggestions need human review",
    }
    atomic_bytes(output / "scan.json", json.dumps(scan, ensure_ascii=False).encode())
    atomic_bytes(
        output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2).encode()
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("fingerprinted", "exact_groups", "near_pairs", "queries")}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection", default="personal")
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    run(args.source_index, args.output, args.collection, args.models)


if __name__ == "__main__":
    main()
