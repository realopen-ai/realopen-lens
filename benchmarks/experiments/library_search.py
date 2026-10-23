"""Partition existing production snapshots and evaluate global retrieval reproducibly.

No source image is copied, modified or re-encoded. Derived generations retain
original encoder specs and selected rows of the verified published matrices.
"""

import argparse
from typing import Any
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np

from benchmarks.metrics.collection import evaluate
from benchmarks.metrics.statistics import distribution
from benchmarks.runtime.resources import MemoryMonitor
from realopen_lens.models.manager import digest
from realopen_lens.organization import duplicates
from realopen_lens.retrieval.library import LibrarySession
from realopen_lens.retrieval.ranking import lexical_index
from realopen_lens.retrieval.session import SearchSession
from realopen_lens.storage.artifacts import atomic_bytes, load_manifest, matrices, save_vectors
from realopen_lens.storage.catalog import Catalog, identifier, local_path


def save(path: Path, payload: dict):
    atomic_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode())


def partition(sources: list[Path], collection: str, output: Path):
    """One deterministic disjoint content partition per source (hash modulo count)."""
    output.mkdir(parents=True, exist_ok=False)
    target = Catalog(output)
    provenance = []
    expected_ids = None
    try:
        for number, source in enumerate(sources):
            original = Catalog(source, readonly=True)
            try:
                found, generation = original.snapshot(collection)
                manifest = load_manifest(original, generation)
                ids = set(manifest["vision_ids"])
                if expected_ids is not None and ids != expected_ids:
                    raise ValueError("Benchmark sources must cover identical image IDs")
                expected_ids = ids
                allowed = {key for key in ids if int(key, 16) % len(sources) == number}
                folder = target.ensure_collection(f"folder-{number+1}", Path(found["root"]))
                job = target.create_job(folder["id"], manifest["config"])
                gen = identifier()
                base = Path("generations") / gen
                directory = output / base
                directory.mkdir(parents=True)
                derived = {
                    **manifest,
                    "images": [r for r in manifest["images"] if r["image_id"] in allowed],
                    "ocr": [r for r in manifest["ocr"] if r["image_id"] in allowed],
                }
                vision, semantic = matrices(original, manifest)
                for name, array, mapping in [
                    ("vision", vision, "vision_ids"),
                    ("semantic", semantic, "passage_ids"),
                ]:
                    mask = [key in allowed for key in manifest[mapping]]
                    derived[mapping] = [key for key in manifest[mapping] if key in allowed]
                    save_vectors(directory / (name + ".npy"), array[mask])
                del vision, semantic
                lexical = lexical_index(derived["ocr"])
                lexical.commit()
                import sqlite3

                with sqlite3.connect(directory / "lexical.sqlite3") as disk:
                    lexical.backup(disk)
                lexical.close()
                for row in derived["images"]:
                    dest = local_path(output, row["thumbnail"])
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copy2(local_path(source, row["thumbnail"]), dest)
                derived["artifacts"] = {
                    name: {"path": (base / file).as_posix(), "sha256": digest(directory / file)}
                    for name, file in [
                        ("vision", "vision.npy"),
                        ("semantic", "semantic.npy"),
                        ("lexical", "lexical.sqlite3"),
                    ]
                }
                save(directory / "manifest.json", derived)
                target.publish(
                    folder["id"],
                    job,
                    gen,
                    (base / "manifest.json").as_posix(),
                    digest(directory / "manifest.json"),
                )
                # Reuse only frozen, validated pixel identities from prior scans.
                cached = duplicates.snapshot(source, manifest)
                if cached:
                    key = duplicates._key(derived)
                    save(
                        output / "duplicates" / (key + ".json"),
                        {
                            "policy": cached["policy"],
                            "key": key,
                            "exact": [],
                            "pixels": {
                                k: v for k, v in cached.get("pixels", {}).items() if k in allowed
                            },
                        },
                    )
                provenance.append(
                    {
                        "source": str(source),
                        "collection": collection,
                        "generation": generation["id"],
                        "manifest_sha256": generation["checksum"],
                        "partition": number,
                        "images": len(allowed),
                    }
                )
            finally:
                original.close()
    finally:
        target.close()
    save(output / "partition.json", {"policy": "sha256-modulo-v1", "sources": provenance})


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    queries = json.loads(args.judgments.read_text(encoding="utf-8"))["queries"]
    if args.limit:
        # Round-robin across EN/FR/AR and content kinds, preserving source order.
        buckets: dict[tuple[str, str], list[dict]] = {}
        for q in queries:
            buckets.setdefault((q["language"], q["kind"]), []).append(q)
        chosen: list[dict] = []
        while len(chosen) < min(args.limit, len(queries)):
            for bucket in buckets.values():
                if bucket and len(chosen) < args.limit:
                    chosen.append(bucket.pop(0))
        queries = chosen
    config = {
        "index": str(args.index.resolve()),
        "collection": args.collection,
        "judgments_sha256": digest(args.judgments),
        "query_ids": [q["id"] for q in queries],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": 2,
        "packages": {n: version(n) for n in ("numpy", "onnxruntime", "tokenizers", "psutil")},
        "sources": {
            n: digest(Path("src/realopen_lens") / n)
            for n in ("retrieval/library.py", "retrieval/ranking.py", "runtime/resident.py")
        },
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Resume configuration changed; use a new output directory")
    save(config_path, config)
    records_path = args.output / "queries.jsonl"
    rows = (
        [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
        if records_path.exists()
        else []
    )
    completed = {r["query_id"] for r in rows}
    session: Any = (
        SearchSession(index_dir=args.index, collection=args.collection, models=args.models)
        if args.collection
        else LibrarySession(index_dir=args.index, models=args.models)
    )
    try:
        with records_path.open("a", encoding="utf-8", newline="\n") as stream:
            for query in queries:
                if query["id"] in completed:
                    continue
                started = perf_counter()
                profile: dict = {}
                trace: dict = {}
                monitor = MemoryMonitor()
                row = {
                    "query_id": query["id"],
                    "language": query["language"],
                    "kind": query["kind"],
                }
                with monitor:
                    try:
                        if args.collection:
                            results = [
                                asdict(r)
                                for r in session.search(
                                    query["text"],
                                    limit=50,
                                    use_feedback=False,
                                    trace=trace,
                                    profile=profile,
                                    use_active_reranker=False,
                                )
                            ]
                        else:
                            payload = session.search(
                                query["text"],
                                limit=50,
                                use_feedback=False,
                                trace=trace,
                                profile=profile,
                            )
                            results = payload["results"]
                        labels = {j["image_sha256"]: j["grade"] for j in query["judgments"]}
                        # A collapsed result represents all exact content aliases.
                        keys = []
                        for r in results:
                            aliases = [r["image_id"], *r.get("duplicate_ids", [])]
                            keys.append(max(aliases, key=lambda k: (labels.get(k, 0), k)))
                        row.update(
                            status="ok",
                            results=results,
                            trace=trace,
                            metrics=evaluate(keys, labels),
                        )
                    except Exception as error:
                        row.update(status="error", error=f"{type(error).__name__}: {error}")
                row.update(
                    seconds=perf_counter() - started,
                    profile=profile,
                    peak_tree_rss_mib=max(monitor.peaks.values(), default=0) / 2**20,
                    memory_samples=monitor.samples,
                    memory_access_errors=monitor.access_errors,
                )
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            "query": query["id"],
                            "status": row["status"],
                            "seconds": round(row["seconds"], 3),
                        }
                    ),
                    flush=True,
                )
    finally:
        session.close()
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {
        "queries": len(rows),
        "failures": len(rows) - len(ok),
        "seconds": distribution([r["seconds"] for r in ok]),
        "peak_tree_rss_mib": distribution([r["peak_tree_rss_mib"] for r in ok]),
        "metrics": {
            name: float(np.mean([r["metrics"][name] for r in ok]))
            for name in ("candidate_recall_at_50", "mrr_at_10", "ndcg_at_10")
        },
    }
    save(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("partition")
    build.add_argument("--source", type=Path, action="append", required=True)
    build.add_argument("--collection", default="personal")
    build.add_argument("--output", type=Path, required=True)
    bench = commands.add_parser("run")
    bench.add_argument("--index", type=Path, required=True)
    bench.add_argument("--collection")
    bench.add_argument("--models", type=Path, default=Path("models"))
    bench.add_argument(
        "--judgments", type=Path, default=Path("benchmarks/sources/retrieval-v2/judgments.json")
    )
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "partition":
        partition(args.source, args.collection, args.output)
    else:
        print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
