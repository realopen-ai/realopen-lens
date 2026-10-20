"""Compare frozen precision policies using cached OCR and complete image embeddings."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.metrics.collection import evaluate
from benchmarks.runtime.resources import MemoryMonitor
from benchmarks.runtime.identity import source_hash
from realopen_lens.models.manager import digest
from realopen_lens.retrieval.ranking import (
    hybrid_rank,
    lexical_frequencies,
    lexical_index,
    rank_lexical,
)
from realopen_lens.storage.artifacts import load_manifest, matrices
from realopen_lens.storage.catalog import Catalog


def vectors(directory: Path, expected: list[dict]) -> tuple[np.ndarray, dict]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if (
        report["errors"]
        or report["inputs"] != len(expected)
        or digest(directory / "vectors.npy") != report["vectors_sha256"]
        or digest(directory / "records.jsonl") != report["records_sha256"]
    ):
        raise ValueError("Incomplete or changed precision probe")
    records = list(
        map(json.loads, (directory / "records.jsonl").read_text(encoding="utf-8").splitlines())
    )
    if [r["id"] for r in records] != [r["id"] for r in expected]:
        raise ValueError("Precision probe input identities differ")
    result = np.load(directory / "vectors.npy", allow_pickle=False)
    if len(result) != len(records) or not np.isfinite(result).all():
        raise ValueError("Invalid precision embeddings")
    return result, report


def run(args) -> dict:
    if args.output.exists():
        raise ValueError("Preserve precision reports; use a new output directory")
    source_root = Path(__file__).resolve().parents[2] / "src/realopen_lens"
    engine_sha256, harness_sha256 = source_hash(source_root), digest(Path(__file__))
    corpus, judgments = load(args.corpus, args.judgments)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    if not inputs.get("full_corpus") or inputs["corpus_sha256"] != digest(args.corpus):
        raise ValueError("Precision retrieval requires the full frozen corpus")
    query_inputs = json.loads(args.query_inputs.read_text(encoding="utf-8"))
    if inputs["queries"] != query_inputs["queries"]:
        raise ValueError("Precision query text inventory changed")
    catalog = Catalog(args.index_dir, readonly=True)
    try:
        _, generation = catalog.snapshot(args.collection)
        manifest = load_manifest(catalog, generation)
        visual, semantic = matrices(catalog, manifest)
    finally:
        catalog.close()
    if inputs["generation"] != generation["id"]:
        raise ValueError("Precision generation changed")
    baseline_config = json.loads((args.baseline_dir / "config.json").read_text(encoding="utf-8"))
    if (
        baseline_config["snapshot"]["generation"] != generation["id"]
        or baseline_config["corpus_sha256"] != digest(args.corpus)
        or baseline_config["judgments_sha256"] != digest(args.judgments)
        or baseline_config["candidate_depths"] != {"visual": 50, "semantic": 50, "lexical": 50}
    ):
        raise ValueError("Precision baseline differs from the frozen evaluation")
    if {r["sha256"] for r in corpus["images"]} != set(manifest["vision_ids"]):
        raise ValueError("Precision corpus differs from the indexed images")
    baseline = {
        r["query_id"]: r
        for r in map(
            json.loads,
            (args.baseline_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    reports: dict = {}
    vision_fp32, reports["fp32-vision"] = vectors(args.fp32_dir / "vision", inputs["images"])
    semantic_fp32, reports["fp32-passages"] = vectors(
        args.fp32_dir / "e5-passage", inputs["passages"]
    )
    query_vectors = {}
    for backend in ("int8", "fp32"):
        for kind in ("siglip2", "e5-query"):
            key = backend + "/" + kind
            query_vectors[key], reports[key] = vectors(
                args.query_probe_dir / backend / kind, inputs["queries"]
            )
    key = "reference/siglip2"
    query_vectors[key], reports[key] = vectors(
        args.query_probe_dir / "reference/siglip2", inputs["queries"]
    )
    for key in ("fp32-vision", "fp32-passages"):
        if reports[key]["identity"]["inputs_sha256"] != digest(args.inputs):
            raise ValueError("Full-corpus probe fingerprint differs")
    for key in query_vectors:
        if reports[key]["identity"]["inputs_sha256"] != digest(args.query_inputs):
            raise ValueError("Query probe fingerprint differs")
    if any(
        report["identity"]["inference_sha256"] != manifest["config"]["spec"]["inference_sha256"]
        for report in reports.values()
    ):
        raise ValueError("Precision preprocessing differs from the indexed collection")
    query_offsets = {row["id"]: i for i, row in enumerate(inputs["queries"])}
    policies = {
        "int8": ("int8", "int8", "int8"),
        "fp32-siglip-vision": ("fp32", "int8", "int8"),
        "fp32-siglip-text": ("int8", "fp32", "int8"),
        "fp32-siglip": ("fp32", "fp32", "int8"),
        "fp32-e5": ("int8", "int8", "fp32"),
        "fp32-all": ("fp32", "fp32", "fp32"),
        "fp32-siglip-native-case": ("fp32", "reference", "int8"),
    }
    args.output.mkdir(parents=True)
    lexical = lexical_index(manifest["ocr"])
    compact = []
    groups: dict[str, list] = defaultdict(list)
    memory = MemoryMonitor()
    try:
        with memory, (args.output / "queries.jsonl").open("w", encoding="utf-8") as stream:
            for query in judgments["queries"]:
                labels = {row["image_sha256"]: row["grade"] for row in query["judgments"]}
                positives = {key for key, grade in labels.items() if grade > 0}
                keywords = rank_lexical(lexical, query["text"], 50)
                stats = lexical_frequencies(lexical, query["text"])
                offset = query_offsets[query["id"]]
                for name, (vision_backend, text_backend, semantic_backend) in policies.items():
                    started = perf_counter()
                    trace: dict = {}
                    vision_ids = (
                        manifest["vision_ids"]
                        if vision_backend == "int8"
                        else [r["id"] for r in inputs["images"]]
                    )
                    passage_ids = (
                        manifest["passage_ids"]
                        if semantic_backend == "int8"
                        else [r["image_id"] for r in inputs["passages"]]
                    )
                    hybrid_rank(
                        visual if vision_backend == "int8" else vision_fp32,
                        query_vectors[text_backend + "/siglip2"][offset],
                        vision_ids,
                        semantic if semantic_backend == "int8" else semantic_fp32,
                        query_vectors[semantic_backend + "/e5-query"][offset],
                        passage_ids,
                        query["text"],
                        manifest["ocr"],
                        lexical_candidates=keywords,
                        lexical_statistics=stats,
                        trace=trace,
                    )
                    if (
                        name == "int8"
                        and trace["rankings"] != baseline[query["id"]]["trace"]["rankings"]
                    ):
                        raise ValueError(
                            "Cached int8 replay differs from production baseline: " + query["id"]
                        )
                    record = {
                        "query_id": query["id"],
                        "policy": name,
                        "language": query["language"],
                        "kind": query.get("kind", "mixed"),
                        "metrics": {
                            mode: evaluate(ranking, labels)
                            for mode, ranking in trace["rankings"].items()
                        },
                        "pool_recall": len(positives & {r["image_id"] for r in trace["candidates"]})
                        / len(positives),
                        "ranking_only_seconds": perf_counter() - started,
                        "positive_ranks": {
                            key: {
                                mode: values.index(key) + 1 if key in values else None
                                for mode, values in trace["rankings"].items()
                            }
                            for key in positives
                        },
                    }
                    stream.write(json.dumps(record | {"trace": trace}, ensure_ascii=False) + "\n")
                    stream.flush()
                    compact.append(record)
                    for group in (
                        "all",
                        "language:" + query["language"],
                        "kind:" + query.get("kind", "mixed"),
                    ):
                        groups[name + "/" + group].append(record)
    finally:
        lexical.close()
    aggregate = {
        key: {
            "queries": len(rows),
            "pool_recall": sum(r["pool_recall"] for r in rows) / len(rows),
            "metrics": {
                mode: {
                    metric: sum(r["metrics"][mode][metric] for r in rows) / len(rows)
                    for metric in rows[0]["metrics"][mode]
                }
                for mode in rows[0]["metrics"]
            },
        }
        for key, rows in groups.items()
    }
    if source_hash(source_root) != engine_sha256 or digest(Path(__file__)) != harness_sha256:
        raise ValueError("Precision evaluation source changed; preserve this attempt and rerun")
    report = {
        "version": 1,
        "generation": generation["id"],
        "inputs_sha256": digest(args.inputs),
        "probe_reports": reports,
        "harness_sha256": harness_sha256,
        "engine_sha256": engine_sha256,
        "aggregate": aggregate,
        "queries": compact,
        "peak_ranking_tree_rss_mib": max(memory.peaks.values(), default=0) / 2**20,
        "limitations": [
            "Previously inspected queries, incomplete judgments; "
            "diagnostic precision comparison only.",
            "Cached OCR and unchanged fusion thresholds isolate precision policies; "
            "thresholds were calibrated for int8 E5.",
            "Ranking-only latency excludes query/image model loading and inference; "
            "use probe records for encoding costs.",
            "FP32 is not enabled in production by this experiment; "
            "int8 and hybrid defaults stay unchanged.",
            "Native-case policy uses official framework text queries and FP32 image embeddings; "
            "it is a separate preprocessing diagnostic, not a pure precision change.",
        ],
    }
    save_json(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "index-dir",
        "inputs",
        "query-inputs",
        "fp32-dir",
        "query-probe-dir",
        "baseline-dir",
        "corpus",
        "judgments",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--collection", required=True)
    report = run(parser.parse_args())
    print(
        json.dumps(
            {key: value for key, value in report["aggregate"].items() if key.endswith("/all")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
