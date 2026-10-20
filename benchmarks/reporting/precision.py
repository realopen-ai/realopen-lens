"""Pair complete production precision runs without mixing collection inventories."""

import argparse
import json
from pathlib import Path
import sqlite3

from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.datasets.downloads import digest
from benchmarks.experiments.saved_precision import compare_controls
from benchmarks.metrics.collection import evaluate
from benchmarks.metrics.statistics import distribution


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def execution_history(directory: Path, records: list[dict]) -> dict:
    """Keep first-pass failures visible even when the latest retry succeeded."""
    with sqlite3.connect(
        (directory / "results.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        latest = {key: json.loads(payload) for key, payload in db.execute("SELECT * FROM queries")}
        attempts = [
            (key, number, json.loads(payload))
            for key, number, payload in db.execute(
                "SELECT query_id,attempt,payload FROM attempts ORDER BY query_id,attempt"
            )
        ]
    if latest != {r["query_id"]: r for r in records}:
        raise ValueError("Retained query exports differ from the result database")
    grouped: dict[str, list] = {}
    for key, number, row in attempts:
        if key not in latest or row["query_id"] != key or row["query"] != latest[key]["query"]:
            raise ValueError("Attempt history differs from frozen queries")
        grouped.setdefault(key, []).append((number, row))
    if set(grouped) != set(latest) or any(
        [n for n, _ in rows] != list(range(1, len(rows) + 1)) or rows[-1][1] != latest[key]
        for key, rows in grouped.items()
    ):
        raise ValueError("Attempt history is incomplete or differs from final results")
    first = [rows[0][1] for rows in grouped.values()]
    quality = [
        (
            row["metrics"]["hybrid"]
            if row["status"] == "ok"
            else evaluate([], {j["image_sha256"]: j["grade"] for j in row["query"]["judgments"]})
        )
        for row in first
    ]
    return {
        "first_pass_queries": len(first),
        "first_pass_successful": sum(r["status"] == "ok" for r in first),
        "first_pass_failed": sum(r["status"] != "ok" for r in first),
        "total_attempts": len(attempts),
        "failed_attempts": [
            {
                "query_id": key,
                "attempt": number,
                **{
                    field: row.get(field)
                    for field in (
                        "error_type",
                        "error",
                        "seconds",
                        "peak_tree_rss_mib",
                        "timestamp",
                    )
                },
            }
            for key, number, row in attempts
            if row["status"] != "ok"
        ],
        "first_pass_deadline_aware_quality": {
            metric: sum(q[metric] for q in quality) / len(quality) for metric in quality[0]
        },
        "first_pass_seconds_including_errors": distribution([r["seconds"] for r in first]),
        "all_attempt_seconds": distribution([r["seconds"] for _, _, r in attempts]),
        "all_attempt_peak_tree_rss_mib": distribution(
            [r["peak_tree_rss_mib"] for _, _, r in attempts]
        ),
        "memory_access_errors": sum(r.get("memory_access_errors", 1) for _, _, r in attempts),
        "memory_complete": all(
            r.get("memory_access_errors", 1) == 0 and r.get("memory_samples", 0) > 0
            for _, _, r in attempts
        ),
    }


def compare(int8: Path, fp32: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Preserve historical comparisons; choose a new output directory")
    configs = [read(p / "config.json") for p in (int8, fp32)]
    a, b = configs
    for key in (
        "corpus_sha256",
        "judgments_sha256",
        "model_registry",
        "threads",
        "packages",
        "engine_sha256",
        "harness_sha256",
        "candidate_depths",
        "search_limit",
        "collection",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Precision runs differ in {key}")
    if any(c.get("reranker_sha256") or c.get("defer_test_metrics") for c in configs):
        raise ValueError("Precision comparisons require fully scored hybrid runs")
    left, right = a["snapshot"], b["snapshot"]
    for key in set(left) | set(right):
        if key not in {"generation", "manifest_sha256", "pipeline", "pipeline_spec"}:
            if left.get(key) != right.get(key):
                raise ValueError(f"Precision collection inventories differ in {key}")
    if (
        left["pipeline_spec"].get("siglip_precision", "int8") != "int8"
        or right["pipeline_spec"].get("siglip_precision", "int8") != "fp32"
        or dict(left["pipeline_spec"], siglip_precision="fp32") != right["pipeline_spec"]
    ):
        raise ValueError("Pipeline specifications must differ only in SigLIP2 precision")
    _, bundle = load(int8 / "corpus.json", int8 / "judgments.json")
    expected = {q["id"]: q for q in bundle["queries"]}
    runs, summaries, histories = [], [], []
    for directory, config in zip((int8, fp32), configs):
        if (
            digest(directory / "corpus.json") != config["corpus_sha256"]
            or digest(directory / "judgments.json") != config["judgments_sha256"]
        ):
            raise ValueError("Retained references differ from the frozen run")
        summary = read(directory / "summary.json")
        records = [
            json.loads(line)
            for line in (directory / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if (
            len(records) != len(expected)
            or {r["query_id"] for r in records} != set(expected)
            or summary["processed_queries"] != len(expected)
            or summary["failed_queries"]
            or summary["quality_deferred_queries"]
            or summary["identity"] != config
            or any(r["status"] != "ok" or r["query"] != expected[r["query_id"]] for r in records)
        ):
            raise ValueError("Precision comparison requires every frozen query successfully scored")
        for metric, value in summary["aggregate"]["hybrid/all"].items():
            if (
                metric != "queries"
                and abs(sum(r["metrics"]["hybrid"][metric] for r in records) / len(records) - value)
                > 1e-12
            ):
                raise ValueError("Precision summary differs from individual query results")
        summaries.append(summary)
        histories.append(execution_history(directory, records))
        runs.append({r["query_id"]: r for r in records})
    pairs = []
    for key, query in expected.items():
        old, new = runs[0][key], runs[1][key]
        positives = {r["image_sha256"] for r in query["judgments"] if r["grade"] > 0}
        pairs.append(
            {
                "query_id": key,
                "text": query["text"],
                "language": query["language"],
                "kind": query.get("kind", "mixed"),
                "family": query["family"],
                "int8": old["metrics"],
                "fp32": new["metrics"],
                "delta_hybrid": {
                    metric: new["metrics"]["hybrid"][metric] - value
                    for metric, value in old["metrics"]["hybrid"].items()
                },
                "positive_ranks": {
                    name: {
                        image: (
                            row["trace"]["rankings"]["hybrid"].index(image) + 1
                            if image in row["trace"]["rankings"]["hybrid"]
                            else None
                        )
                        for image in sorted(positives)
                    }
                    for name, row in (("int8", old), ("fp32", new))
                },
            }
        )
    memory_errors = [h["memory_access_errors"] for h in histories]
    controls = compare_controls(Path(b["index_dir"]), Path(a["index_dir"]), a["collection"])
    if (
        controls["source_generation"] != right["generation"]
        or controls["target_generation"] != left["generation"]
    ):
        raise ValueError("Controlled collections changed since evaluation")
    report = {
        "queries": len(expected),
        "reference_images": len(read(int8 / "corpus.json")["images"]),
        "collection_images": left["unique_images"],
        "additional_unreviewed_images": left.get("additional_unreviewed_images", 0),
        "memory_complete": all(value == 0 for value in memory_errors)
        and all(h["memory_complete"] for h in histories),
        "memory_access_errors": dict(zip(("int8", "fp32"), memory_errors)),
        "policies": dict(zip(("int8", "fp32"), summaries)),
        "execution_history": dict(zip(("int8", "fp32"), histories)),
        "reporter_sha256": digest(Path(__file__)),
        "paired_queries": pairs,
        "controls": controls,
        "ndcg_regressions": [r["query_id"] for r in pairs if r["delta_hybrid"]["ndcg_at_10"] < 0],
        "limitations": (
            "Known, previously inspected judgments; additional collection images have no invented "
            "labels or provenance. Changed collection size prevents direct comparison "
            "with older runs."
        ),
    }
    output.mkdir(parents=True)
    save_json(output / "comparison.json", report)
    lines = [
        "# Saved collection precision comparison",
        "",
        "| Policy | Recall@50 | MRR@10 | nDCG@10 | Final-success median s | P95 s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["policies"].items():
        quality, times = summary["aggregate"]["hybrid/all"], summary["hybrid_end_to_end_seconds"]
        lines.append(
            f"| {name} | {quality['candidate_recall_at_50']:.4f} | {quality['mrr_at_10']:.4f} | "
            f"{quality['ndcg_at_10']:.4f} | {times['median']:.3f} | {times['p95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Quality uses final successful queries; retries do not erase initial failures.",
            "",
            "| Policy | First-pass successes | Failures | Total attempts |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, history in report["execution_history"].items():
        lines.append(
            f"| {name} | {history['first_pass_successful']} | "
            f"{history['first_pass_failed']} | {history['total_attempts']} |"
        )
    lines.extend(
        [
            "",
            "First-pass latency including failures, deadline-aware quality, and all "
            "failed attempts are retained in comparison.json.",
            "",
            "| Policy | First-pass median s | P95 s | P99 s | Max s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, history in report["execution_history"].items():
        times = history["first_pass_seconds_including_errors"]
        lines.append(
            f"| {name} | {times['median']:.3f} | {times['p95']:.3f} | "
            f"{times['p99']:.3f} | {times['max']:.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.int8, args.fp32, args.output)


if __name__ == "__main__":
    main()
