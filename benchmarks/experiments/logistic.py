"""Train on reviewed candidate pairs, select on validation, then freeze a test comparison."""

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load as load_judgments
from benchmarks.datasets.training_review import apply
from benchmarks.experiments.logistic_fit import fit
from benchmarks.metrics.collection import evaluate
from benchmarks.metrics.statistics import distribution
from benchmarks.reporting.collections import candidate_rows
from benchmarks.runtime.identity import source_hash
from benchmarks.runtime.resources import MemoryMonitor
from realopen_lens.learning.logistic import load as load_model

L2_GRID = (0.001, 0.01, 0.1, 1.0)
METRICS = ("candidate_recall_at_50", "mrr_at_10", "ndcg_at_10", "judged_fraction_at_10")


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_records(records: list[dict], bundle: dict, corpus: dict, config: dict) -> None:
    expected = {q["id"]: q for q in bundle["queries"]}
    keys = [r["query_id"] for r in records]
    images = {r["sha256"] for r in corpus["images"]}
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("Replay requires exactly one record for every frozen query")
    if config["candidate_feature_schema"] != 1 or config["search_limit"] != 50:
        raise ValueError("Replay requires the version-1 top-50 production trace schema")
    spec = config.get("snapshot", {}).get("pipeline_spec", {})
    if spec.get("siglip_precision", "int8") != "int8":
        raise ValueError("Existing reranker training requires INT8 SigLIP2 compatibility")
    budget = {name: 50 for name in ("visual", "semantic", "lexical")}
    if config.get("candidate_depths", budget) != budget:
        raise ValueError("Reranker training requires the original 50 candidates per channel")
    for row in records:
        if row["status"] != "ok" or row["query"] != expected[row["query_id"]]:
            raise ValueError("Replay contains errors or different frozen query references")
        trace = row["trace"]
        candidates = [r["image_id"] for r in trace["candidates"]]
        if (
            trace.get("candidate_depths", budget) != budget
            or len(candidates) > 150
            or any(len(trace["rankings"].get(name, [])) > 50 for name in budget)
        ):
            raise ValueError("Reranker training requires the original 50 candidates per channel")
        if (
            trace["generation"] != config["snapshot"]["generation"]
            or trace["searchable_images"] != len(images)
            or len(candidates) != len(set(candidates))
            or not set(candidates) <= images
            or trace["rankings"]["hybrid"] != candidates[:50]
        ):
            raise ValueError("Replay candidate identity or snapshot is inconsistent")


def quality(ranking: list[str], query: dict) -> dict:
    return evaluate(ranking, {r["image_sha256"]: r["grade"] for r in query["judgments"]})


def mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Quality aggregation needs evaluated queries")
    return {key: float(np.mean([r[key] for r in rows])) for key in METRICS}


def run(args) -> dict:
    corpus, original = load_judgments(args.corpus, args.judgments)
    review = json.loads(args.review.read_text(encoding="utf-8"))
    baseline = args.baseline_dir
    config = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    if any(
        source[key] != digest(path)
        for source in (review, config)
        for key, path in (("corpus_sha256", args.corpus), ("judgments_sha256", args.judgments))
    ):
        raise ValueError("Review or baseline is bound to different frozen inputs")
    records = read_records(baseline / "queries.jsonl")
    validate_records(records, original, corpus, config)
    reviewed = apply(original, corpus, review)
    queries = {row["id"]: row for row in reviewed["queries"]}
    images = {row["sha256"]: row for row in corpus["images"]}
    training = [
        candidate
        for row in records
        if queries[row["query_id"]]["split"] == "train"
        for candidate in candidate_rows(row, queries[row["query_id"]], images)
        if candidate["training_eligible"]
    ]
    validation = [row for row in records if queries[row["query_id"]]["split"] == "validation"]
    if not validation:
        raise ValueError("Model selection requires validation queries")
    if args.output.exists():
        raise ValueError("Use a new output directory to preserve previous training experiments")
    args.output.mkdir(parents=True)
    identity = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_sha256": digest(args.corpus),
        "judgments_sha256": digest(args.judgments),
        "review_sha256": digest(args.review),
        "baseline_queries_sha256": digest(baseline / "queries.jsonl"),
        "baseline_config_sha256": digest(baseline / "config.json"),
        "baseline_identity": config,
        "l2_grid": list(L2_GRID),
        "selection": "validation nDCG@10, then MRR@10, then stronger L2; test read after freeze",
        "engine_sha256": source_hash(Path(__file__).resolve().parents[2] / "src/realopen_lens"),
        "harness_sha256": source_hash(Path(__file__).resolve().parents[1]),
        "environment": {"platform": platform.platform(), "numpy": np.__version__},
    }
    save_json(args.output / "config.json", identity)
    save_json(args.output / "judgments.json", reviewed)
    shutil.copyfile(args.review, args.output / "training-review.json")
    root = Path(__file__).resolve().parents[2]
    for source, name in (
        (root / "src/realopen_lens", "engine-source"),
        (root / "benchmarks", "harness-source"),
    ):
        shutil.copytree(source, args.output / name, ignore=shutil.ignore_patterns("__pycache__"))
    with (args.output / "training-train.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in training:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    spec = config["snapshot"]["pipeline_spec"]
    compatibility = {key: spec[key] for key in ("model_registry", "inference_sha256")}
    memory = MemoryMonitor()
    memory.phase = "training_and_replay"
    started = perf_counter()
    trials = []
    with memory:
        for l2 in L2_GRID:
            artifact = fit(training, l2, compatibility)
            artifact["provenance"] = {
                key: identity[key]
                for key in (
                    "corpus_sha256",
                    "judgments_sha256",
                    "review_sha256",
                    "baseline_queries_sha256",
                    "engine_sha256",
                    "harness_sha256",
                )
            }
            path = args.output / f"trial-l2-{l2:g}.json"
            save_json(path, artifact)
            model = load_model(path)
            metrics = mean_metrics(
                [
                    quality(model.rank(row["trace"]["candidates"])[:50], queries[row["query_id"]])
                    for row in validation
                ]
            )
            trials.append({"l2": l2, "validation": metrics, "model": path.name})
        chosen = max(
            trials,
            key=lambda t: (t["validation"]["ndcg_at_10"], t["validation"]["mrr_at_10"], t["l2"]),
        )
        shutil.copyfile(args.output / chosen["model"], args.output / "model.json")
        model = load_model(args.output / "model.json")
        save_json(
            args.output / "selection.json",
            {"trials": trials, "selected": chosen, "frozen_model_sha256": model.sha256},
        )
        # Only the selected, already frozen model is evaluated against test labels.
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        results = []
        timings = []
        for row in records:
            query = queries[row["query_id"]]
            candidates = row["trace"]["candidates"]
            ranking = model.rank(candidates)
            durations = []
            for _ in range(10):
                start = perf_counter()
                model.rank(candidates)
                durations.append((perf_counter() - start) * 1000)
            timings.append(float(np.median(durations)))
            metrics = {
                "hybrid": quality(row["trace"]["rankings"]["hybrid"], query),
                "logistic": quality(ranking[:50], query),
            }
            for tag in (
                "all",
                "split:" + query["split"],
                "language:" + query["language"],
                "kind:" + query.get("kind", "mixed"),
                "exposure:" + query.get("source_exposure", "unspecified"),
            ):
                for mode, values in metrics.items():
                    groups[tag][mode].append(values)
            positives = {r["image_sha256"] for r in query["judgments"] if r["grade"] > 0}
            results.append(
                {
                    "query_id": query["id"],
                    "query": query,
                    "rankings": {"hybrid": row["trace"]["rankings"]["hybrid"], "logistic": ranking},
                    "metrics": metrics,
                    "pool_recall": len(set(ranking) & positives) / len(positives),
                    "rerank_median_ms": timings[-1],
                    "scores": {
                        candidate["image_id"]: float(score)
                        for candidate, score in zip(candidates, model.scores(candidates))
                    },
                }
            )
    summary = {
        "version": 1,
        "queries": len(results),
        "model_sha256": model.sha256,
        "selected_l2": chosen["l2"],
        "training_pairs": len(training),
        "training_positives": sum(row["label"] for row in training),
        "review_decisions": dict(Counter(str(row["grade"]) for row in review["pairs"])),
        "groups": {
            tag: {
                mode: {"queries": len(values), **mean_metrics(values)}
                for mode, values in modes.items()
            }
            for tag, modes in groups.items()
        },
        "rerank_only_median_per_query_ms": distribution(timings),
        "training_and_replay_seconds": perf_counter() - started,
        "sampled_process_tree_peak_mib": max(memory.peaks.values(), default=0) / 2**20,
        "memory_samples": memory.samples,
        "model_bytes": (args.output / "model.json").stat().st_size,
        "parameter_array_bytes": sum(
            array.nbytes for array in (model.mean, model.scale, model.weights)
        ),
        "pool_recall_by_split": {
            split: float(
                np.mean([r["pool_recall"] for r in results if r["query"]["split"] == split])
            )
            for split in ("train", "validation", "test")
        },
        "limitations": [
            "Candidate replay isolates ranking cost; it is not full search latency.",
            "Known-judgment metrics; unjudged candidates may be relevant.",
            "Agent review only, no independent human review.",
            "Test was previously reported for hybrid and includes a corrected test intent.",
            "Only 34 original positive source components; validation lacks text-heavy sources.",
        ],
    }
    with (args.output / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(args.output / "summary.json", summary)
    print(
        json.dumps({k: summary[k] for k in ("queries", "selected_l2", "training_pairs")}),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
