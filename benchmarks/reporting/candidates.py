"""Paired candidate-depth comparisons from retained production query records."""

import json
from pathlib import Path

from benchmarks.datasets.photos import save_json


def compare(directories: list[Path], output: Path) -> dict:
    runs = []
    identity = None
    query_ids = None
    baseline = None
    pairs = []
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        config = summary["identity"]
        comparable = {key: value for key, value in config.items() if key != "candidate_depths"}
        if identity is not None and comparable != identity:
            raise ValueError("Candidate runs differ beyond their candidate depths")
        identity = comparable
        records = {
            row["query_id"]: row
            for row in map(
                json.loads, (directory / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            )
        }
        if (
            summary["failed_queries"]
            or summary["processed_queries"] != summary["expected_queries"]
            or summary["quality_deferred_queries"]
            or len(records) != summary["expected_queries"]
        ):
            raise ValueError("Depth comparison requires complete, successful, scored query runs")
        if query_ids is not None and records.keys() != query_ids:
            raise ValueError("Candidate runs have different query inventories")
        query_ids = records.keys()
        if baseline is None:
            baseline = records
        runs.append(
            {
                "directory": str(directory),
                "candidate_depths": config["candidate_depths"],
                "aggregate": summary["aggregate"],
                **{
                    key: summary[key]
                    for key in (
                        "candidate_pool_recall",
                        "candidate_pool_recall_by_group",
                        "candidate_pool_size",
                        "hybrid_end_to_end_seconds",
                        "peak_tree_rss_mib",
                    )
                },
            }
        )
        for key, row in records.items():
            before = baseline[key]
            labels = {r["image_sha256"]: r["grade"] for r in row["query"]["judgments"]}
            positives = {image for image, grade in labels.items() if grade > 0}
            pool = {r["image_id"] for r in row["trace"]["candidates"]}
            old_pool = {r["image_id"] for r in before["trace"]["candidates"]}
            pairs.append(
                {
                    "query_id": key,
                    "candidate_depths": config["candidate_depths"],
                    "language": row["query"]["language"],
                    "kind": row["query"].get("kind", "mixed"),
                    "pool_size": len(pool),
                    "pool_recall": row["pool_recall"],
                    "new_pool_positives": sorted((pool - old_pool) & positives),
                    "lost_pool_positives": sorted((old_pool - pool) & positives),
                    "hybrid_metrics": row["metrics"]["hybrid"],
                    "hybrid_delta": {
                        metric: row["metrics"]["hybrid"][metric]
                        - before["metrics"]["hybrid"][metric]
                        for metric in ("candidate_recall_at_50", "mrr_at_10", "ndcg_at_10")
                    },
                    "seconds": row["seconds"],
                    "peak_tree_rss_mib": row["peak_tree_rss_mib"],
                }
            )
    if not runs:
        raise ValueError("No candidate runs supplied")
    report: dict = {
        "version": 1,
        "identity": identity,
        "runs": runs,
        "paired_queries": pairs,
        "limitations": [
            "Previously inspected multilingual queries are diagnostic, not a new holdout.",
            "Known judgments are incomplete; pool recovery is not final ranking success.",
            "Sequential full-request timings include inference and OS cache effects.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "comparison.json", report)
    lines = [
        "# Candidate-depth diagnostic study",
        "",
        "| Visual / semantic / lexical | Pool recall | Recall@50 | MRR@10 | nDCG@10 | "
        "Median s | P95 s | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        depths = "/".join(
            str(run["candidate_depths"][k]) for k in ("visual", "semantic", "lexical")
        )
        quality = run["aggregate"]["hybrid/all"]
        latency = run["hybrid_end_to_end_seconds"]
        lines.append(
            f"| {depths} | {run['candidate_pool_recall']['mean']:.4f} | "
            f"{quality['candidate_recall_at_50']:.4f} | {quality['mrr_at_10']:.4f} | "
            f"{quality['ndcg_at_10']:.4f} | {latency['median']:.3f} | "
            f"{latency['p95']:.3f} | {run['peak_tree_rss_mib']['max']:.1f} |"
        )
    lines.extend(["", *["- " + text for text in report["limitations"]]])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
