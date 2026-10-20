"""Diagnostic original+reviewed-English expansion using retained production requests.

This measures a translation oracle, not an automatic translator. No query is
rewritten using its relevance labels and no new production default is selected.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.metrics.collection import evaluate
from benchmarks.metrics.statistics import distribution


def expand(original: dict, english: dict) -> dict:
    query, translation = original["query"], english["query"]
    if (
        original["status"] != "ok"
        or english["status"] != "ok"
        or query["family"] != translation["family"]
        or translation["language"] != "en"
        or query["split"] != translation["split"]
    ):
        raise ValueError(
            "Expansion requires successful, reviewed translations of the same family/fold"
        )
    # One translated variant maximum; a query already in English has one request.
    variants = [original] if query["id"] == translation["id"] else [original, english]
    scores: dict[str, float] = {}
    channels: dict[str, dict[str, float]] = {key: {} for key in ("visual", "semantic", "lexical")}
    for variant in variants:
        for candidate in variant["trace"]["candidates"]:
            key = candidate["image_id"]
            scores[key] = max(scores.get(key, -float("inf")), candidate["hybrid_score"])
        for channel, values in channels.items():
            for rank, key in enumerate(variant["trace"]["rankings"][channel], 1):
                values[key] = max(values.get(key, 0), 1 / (60 + rank))
    # Correlated language variants do not get additional votes within a channel.
    rrf = {key: sum(values.get(key, 0) for values in channels.values()) for key in scores}
    original_rrf: dict[str, float] = defaultdict(float)
    for channel in ("visual", "semantic", "lexical"):
        for rank, key in enumerate(original["trace"]["rankings"][channel], 1):
            original_rrf[key] += 1 / (60 + rank)
    rankings = {
        "original": original["trace"]["rankings"]["hybrid"],
        "max_hybrid": sorted(scores, key=lambda key: (-scores[key], key))[:50],
        "rrf_original": sorted(original_rrf, key=lambda key: (-original_rrf[key], key))[:50],
        "rrf_max_per_channel": sorted(rrf, key=lambda key: (-rrf[key], key))[:50],
    }
    labels = {row["image_sha256"]: row["grade"] for row in query["judgments"]}
    positives = {key for key, grade in labels.items() if grade > 0}
    old = {r["image_id"] for r in original["trace"]["candidates"]}
    return {
        "query_id": query["id"],
        "language": query["language"],
        "family": query["family"],
        "kind": query.get("kind", "mixed"),
        "variant_ids": [r["query_id"] for r in variants],
        "pool_size": len(scores),
        "original_pool_size": len(old),
        "pool_recall": len(positives & scores.keys()) / len(positives),
        "original_pool_recall": len(positives & old) / len(positives),
        "new_pool_positives": sorted((scores.keys() - old) & positives),
        "rankings": rankings,
        "metrics": {k: evaluate(v, labels) for k, v in rankings.items()},
        "reused_request_seconds": sum(r["seconds"] for r in variants),
    }


def run(baseline_dir: Path, output: Path) -> dict:
    config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    _, bundle = load(baseline_dir / "corpus.json", baseline_dir / "judgments.json")
    if config.get("candidate_depths") != {"visual": 50, "semantic": 50, "lexical": 50}:
        raise ValueError("Expansion experiment requires 50 candidates per original channel")
    records = {
        row["query_id"]: row
        for row in map(
            json.loads, (baseline_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    if set(records) != {q["id"] for q in bundle["queries"]}:
        raise ValueError("Expansion requires the complete frozen query inventory")
    english = {}
    for record in records.values():
        query = record["query"]
        if query["language"] == "en":
            if query["family"] in english:
                raise ValueError("Ambiguous English translation")
            english[query["family"]] = record
    rows = []
    groups: dict[str, list] = defaultdict(list)
    for key, record in sorted(records.items()):
        started = perf_counter()
        row = expand(record, english[record["query"]["family"]])
        row["fusion_only_seconds"] = perf_counter() - started
        rows.append(row)
        for group in ("all", "language:" + row["language"], "kind:" + row["kind"]):
            groups[group].append(row)
    aggregate = {}
    for group, values in groups.items():
        aggregate[group] = {
            "queries": len(values),
            "original_pool_recall": sum(r["original_pool_recall"] for r in values) / len(values),
            "expanded_pool_recall": sum(r["pool_recall"] for r in values) / len(values),
            "pool_size": distribution([r["pool_size"] for r in values]),
            "metrics": {
                mode: {
                    metric: sum(r["metrics"][mode][metric] for r in values) / len(values)
                    for metric in values[0]["metrics"][mode]
                }
                for mode in values[0]["metrics"]
            },
        }
    report = {
        "version": 1,
        "baseline_identity": config,
        "aggregate": aggregate,
        "queries": rows,
        "limitations": [
            "Reviewed English translations are an oracle; no automatic local translator evaluated.",
            "All query families were previously inspected; results are diagnostic.",
            "Expansion uses up to twice the candidate channels and query encodings.",
            "Reused request times are not measured expansion latency or translation cost.",
            "Raw hybrid scores across languages are not calibrated; max fusion is experimental.",
            "RRF expansion must be compared with its original-query RRF control, not only hybrid.",
        ],
    }
    if output.exists():
        raise ValueError("Preserve previous expansion results; choose a new output directory")
    output.mkdir(parents=True)
    save_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.baseline_dir, args.output)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
