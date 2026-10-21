"""Evaluate reviewed queries against an immutable production collection generation."""

import argparse
from datetime import UTC, datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sqlite3
import sys
import shutil
import realopen_lens
from time import perf_counter
from typing import Any

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.metrics.collection import evaluate
from benchmarks.runtime.collections import snapshot
from benchmarks.reporting.collections import export
from benchmarks.runtime.identity import source_hash
from benchmarks.runtime.resources import MemoryMonitor, write_resources
from realopen_lens.models.manager import registry, verify
from realopen_lens.retrieval.search import search_collection
from realopen_lens.retrieval.candidates import CandidateDepths
from realopen_lens.runtime.offline import enforce
from realopen_lens.runtime.resident import ResidentQueryEncoder


def run(args):
    if getattr(args, "resident", False):
        with ResidentQueryEncoder() as encoder:
            return _run(args, encoder)
    return _run(args)


def _run(args, encoder: ResidentQueryEncoder | None = None):
    encoder_options: dict[str, Any] = {"query_encoder": encoder} if encoder is not None else {}
    corpus, bundle = load(args.corpus, args.judgments)

    def current_snapshot(*, verify_files: bool = True) -> dict:
        return snapshot(
            args.index_dir,
            args.collection,
            corpus,
            verify_files=verify_files,
            corpus_prefix=getattr(args, "corpus_prefix", Path(".")),
            allow_extra_images=getattr(args, "allow_extra_images", False),
        )

    state = current_snapshot()
    verify(args.models, "int8", state["pipeline_spec"].get("siglip_precision", "int8"))
    reranker = getattr(args, "reranker", None)
    reranker_sha256 = digest(reranker) if reranker is not None else None
    defer_test = getattr(args, "defer_test_metrics", False)
    depths = CandidateDepths(
        **{
            name: getattr(args, name + "_candidates", 50)
            for name in ("visual", "semantic", "lexical")
        }
    )
    if reranker is not None and depths != CandidateDepths():
        raise ValueError("Existing reranker models require 50 candidates per channel")
    root = Path(__file__).resolve().parents[2]
    source_root = root / "src/realopen_lens"
    installed_root = Path(realopen_lens.__file__).parent
    if source_hash(source_root) != source_hash(installed_root):
        raise ValueError("Installed engine differs from source; reinstall realopen-lens first")
    identity = {
        "version": 1,
        "snapshot": state,
        "index_dir": str(args.index_dir.resolve()),
        "collection": args.collection,
        "corpus_sha256": digest(args.corpus),
        "judgments_sha256": digest(args.judgments),
        "models": str(args.models.resolve()),
        "model_registry": registry(),
        "threads": args.threads,
        "search_limit": 50,
        "candidate_feature_schema": 1,
        "candidate_depths": depths.as_dict(),
        "engine_sha256": source_hash(source_root),
        "harness_sha256": source_hash(root / "benchmarks"),
        "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version},
        "packages": {
            key: version(key) for key in ("numpy", "onnxruntime", "tokenizers", "Pillow", "psutil")
        },
    }
    if reranker is not None:
        identity["reranker_sha256"] = reranker_sha256
    if encoder is not None:
        identity["query_runtime"] = "resident"
    if defer_test:
        identity["defer_test_metrics"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "config.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != identity:
        raise ValueError("Resume identity changed; use a new output directory")
    save_json(config, identity)
    save_json(args.output / "corpus.json", corpus)
    save_json(args.output / "judgments.json", bundle)
    if reranker is not None:
        shutil.copyfile(reranker, args.output / "reranker.json")
    for source, name in ((source_root, "engine-source"), (root / "benchmarks", "harness-source")):
        if not (args.output / name).exists():
            shutil.copytree(
                source, args.output / name, ignore=shutil.ignore_patterns("__pycache__")
            )
    db = sqlite3.connect(args.output / "results.sqlite")
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS queries (query_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS attempts (query_id TEXT NOT NULL, "
            "attempt INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(query_id, attempt))"
        )
        done = {
            key
            for key, payload in db.execute("SELECT query_id,payload FROM queries")
            if not args.retry_errors or json.loads(payload)["status"] != "error"
        }
        for query in bundle["queries"]:
            if query["id"] in done:
                continue
            # Refuse a moving target rather than mixing generations or source inventories.
            if current_snapshot(verify_files=False) != state:
                raise ValueError("Collection changed during evaluation")
            if (
                source_hash(source_root) != identity["engine_sha256"]
                or source_hash(root / "benchmarks") != identity["harness_sha256"]
            ):
                raise ValueError("Source changed during evaluation")
            if reranker is not None and digest(reranker) != reranker_sha256:
                raise ValueError("Reranker changed during evaluation")
            memory = MemoryMonitor()
            memory.phase = "hybrid_query"
            started = perf_counter()
            record = {
                "query_id": query["id"],
                "query": query,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            try:
                with memory:
                    trace: dict = {}
                    results = search_collection(
                        query["text"],
                        use_feedback=False,
                        collection=args.collection,
                        index_dir=args.index_dir,
                        models=args.models,
                        limit=50,
                        threads=args.threads,
                        trace=trace,
                        candidate_depths=depths,
                        **({"reranker": reranker} if reranker is not None else {}),
                        **encoder_options,
                    )
                    if (
                        trace["generation"] != state["generation"]
                        or trace["searchable_images"] != state["unique_images"]
                    ):
                        raise ValueError("Search observed a changed snapshot")
                    mode = "logistic" if reranker is not None else "hybrid"
                    if reranker is not None and trace.get("reranker_sha256") != reranker_sha256:
                        raise ValueError("Search observed a changed reranker")
                    if [r.image_id for r in results] != trace["rankings"][mode]:
                        raise ValueError("Candidate trace differs from returned ranking")
                    record.update(status="ok", trace=trace, metrics={})
                    if defer_test and query.get("study_role", query["split"]) == "test":
                        # Capture production candidates without scoring the fresh holdout.
                        # The study scores it only after freezing validation selection.
                        record["quality_deferred"] = True
                    else:
                        labels = {r["image_sha256"]: r["grade"] for r in query["judgments"]}
                        positives = {key for key, value in labels.items() if value > 0}
                        record.update(
                            metrics={
                                mode: evaluate(ranking, labels)
                                for mode, ranking in trace["rankings"].items()
                            },
                            pool_recall=len(
                                {r["image_id"] for r in trace["candidates"]} & positives
                            )
                            / len(positives),
                            missed_pool_positives=sorted(
                                positives - {r["image_id"] for r in trace["candidates"]}
                            ),
                            missed_hybrid_top50_positives=sorted(
                                positives - set(trace["rankings"]["hybrid"])
                            ),
                        )
            except Exception as error:
                record.update(
                    status="error", error_type=type(error).__name__, error=str(error)[:2000]
                )
            record.update(
                seconds=perf_counter() - started,
                peak_tree_rss_mib=max(memory.peaks.values(), default=0) / 2**20,
                memory_samples=memory.samples,
                memory_access_errors=memory.access_errors,
            )
            with db:
                payload = json.dumps(record, ensure_ascii=False)
                attempt = db.execute(
                    "SELECT COALESCE(MAX(attempt),0)+1 FROM attempts WHERE query_id=?",
                    (query["id"],),
                ).fetchone()[0]
                db.execute("INSERT INTO attempts VALUES (?, ?, ?)", (query["id"], attempt, payload))
                db.execute("INSERT OR REPLACE INTO queries VALUES (?, ?)", (query["id"], payload))
            write_resources(args.output, memory)
            print(f"{query['id']}: {record['status']} ({record['seconds']:.2f}s)", flush=True)
        if current_snapshot() != state:
            raise ValueError("Collection changed during evaluation")
        if reranker is not None and digest(reranker) != reranker_sha256:
            raise ValueError("Reranker changed during evaluation")
        summary = export(args.output, db, corpus, bundle, identity)
        if summary["failed_queries"]:
            raise ValueError(f"{summary['failed_queries']} queries failed; see retained records")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--collection", default="retrieval-evaluation")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--corpus-prefix", type=Path, default=Path("."))
    parser.add_argument(
        "--allow-extra-images",
        action="store_true",
        help="Keep additional collection images as explicitly unreviewed distractors",
    )
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--resident", action="store_true", help="Reuse owned query encoders")
    parser.add_argument(
        "--defer-test-metrics",
        action="store_true",
        help="Capture fresh test candidates without scoring before model freeze",
    )
    parser.add_argument("--reranker", type=Path, help="Experimental local logistic JSON model")
    for name in ("visual", "semantic", "lexical"):
        parser.add_argument(
            "--" + name + "-candidates", type=int, choices=range(1, 1001), default=50
        )
    args = parser.parse_args()
    enforce()
    run(args)


if __name__ == "__main__":
    main()
