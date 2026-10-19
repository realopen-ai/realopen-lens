"""Compare retained runs without replacing their individual records or judgments."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.suite import distribution


def read_records(path: Path) -> dict:
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def compare(baseline: Path, final: Path, output: Path) -> dict:
    before = json.loads((baseline / "summary.json").read_text())
    after = json.loads((final / "summary.json").read_text())
    for key in [
        "manifest_sha256",
        "queries",
        "artifacts",
        "languages",
        "threads",
        "packages",
        "host",
    ]:
        if before["config"][key] != after["config"][key]:
            raise ValueError(f"Runs are not comparable: {key} differs")
    if before["config"].get("evaluation_protocol_sha256") != after["config"].get(
        "evaluation_protocol_sha256"
    ):
        raise ValueError("Runs use different reference corrections")
    old = read_records(baseline / "individual.jsonl")
    new = read_records(final / "individual.jsonl")
    if set(old) != set(new):
        raise ValueError("Runs have different image sets")

    def ocr_environment(records: dict) -> set[str]:
        return {
            json.dumps(
                {key: row["stages"]["ocr"].get(key) for key in ("version", "language_pack_sha256")},
                sort_keys=True,
            )
            for row in records.values()
            if row["stages"]["ocr"]["status"] == "ok"
        }

    if ocr_environment(old) != ocr_environment(new):
        raise ValueError("Tesseract version or selected language-pack hashes differ")
    changes = []
    for image_id, row in sorted(new.items()):
        previous = old[image_id]
        changes.append(
            {
                "id": image_id,
                "category": row["category"],
                "split": row["split"],
                "baseline_status": previous["status"],
                "final_status": row["status"],
                "baseline_quality": previous["quality"],
                "final_quality": row["quality"],
                "quality_change": previous["quality_status"] + " -> " + row["quality_status"],
                "latency_change_seconds": row["latency_seconds"] - previous["latency_seconds"],
                "baseline_latency_seconds": previous["latency_seconds"],
                "final_latency_seconds": row["latency_seconds"],
            }
        )
    oldq = read_records(baseline / "queries.jsonl")
    newq = read_records(final / "queries.jsonl")
    query_changes = []
    groups: dict[str, list] = defaultdict(list)
    for key, row in sorted(newq.items()):
        previous = oldq[key]
        change = {
            "id": key,
            "text": row["text"],
            "category": row["category"],
            "language": row["language"],
            "kind": row["kind"],
            "split": row["split"],
            "relevant": row["relevant"],
            "baseline": previous["metrics"],
            "final": row["metrics"],
            "baseline_rankings": previous["rankings"],
            "final_rankings": row["rankings"],
        }
        query_changes.append(change)
        for label in [
            "all",
            row["category"],
            "split:" + row["split"],
            "language:" + row["language"],
            "kind:" + row["kind"],
            row["category"] + ":" + row["split"],
            row["language"] + ":" + row["kind"],
        ]:
            groups[label].append(change)
    retrieval: dict[str, Any] = {}
    for group, items in sorted(groups.items()):
        retrieval[group] = {"queries": len(items)}
        for channel in ["visual", "lexical", "semantic_ocr", "hybrid"]:
            retrieval[group][channel] = {}
            for metric in [
                "recall_at_5",
                "hit_rate_at_5",
                "mrr_at_10",
                "ndcg_at_10",
                "candidate_recall_at_50",
            ]:
                a = float(np.mean([r["baseline"][channel][metric] for r in items]))
                b = float(np.mean([r["final"][channel][metric] for r in items]))
                retrieval[group][channel][metric] = {"baseline": a, "final": b, "delta": b - a}
    output.mkdir(parents=True, exist_ok=True)
    (output / "image-changes.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in changes)
    )
    (output / "query-changes.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in query_changes)
    )
    regressions = [
        r
        for r in query_changes
        if r["final"]["hybrid"]["mrr_at_10"] < r["baseline"]["hybrid"]["mrr_at_10"]
    ]
    # Source-stratified OCR prevents authored templates from hiding real-image weaknesses.
    strata: dict[str, Any] = {}
    for kind in ["real", "synthetic", "derived"]:
        strata[kind] = {}
        for label, records in [("baseline", old), ("final", new)]:
            items = [r for r in records.values() if r["source"]["kind"] == kind]
            scored = [r for r in items if r["quality"]]
            strata[kind][label] = {
                "count": len(items),
                "quality_evaluated": len(scored),
                "quality_passed": sum(r["quality_status"] == "pass" for r in scored),
                "latency_seconds": distribution([r["latency_seconds"] for r in items]),
            }
    result = {
        "baseline": str(baseline),
        "final": str(final),
        "images": len(old),
        "queries": len(oldq),
        "dataset_sha256": before["config"]["manifest_sha256"],
        "retrieval": retrieval,
        "quality_changes": dict(Counter(r["quality_change"] for r in changes)),
        "ocr_by_source": strata,
        "retrieval_regressions": regressions,
        "quality_regressions": [r for r in changes if r["quality_change"] == "pass -> fail"],
        "metric_regressions": [
            r
            for r in changes
            if any(
                r["baseline_quality"].get(key) is not None
                and r["final_quality"].get(key) is not None
                and r["baseline_quality"][key] - r["final_quality"][key] > 0.05
                for key in ("token_recall", "token_precision", "key_phrase_recall")
            )
        ],
        "baseline_aggregate": before["aggregate"],
        "final_aggregate": after["aggregate"],
    }
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.final, args.output)
    print(
        json.dumps(
            {
                "quality_changes": result["quality_changes"],
                "retrieval": result["retrieval"]["all"],
                "quality_regressions": len(result["quality_regressions"]),
                "retrieval_regressions": len(result["retrieval_regressions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
