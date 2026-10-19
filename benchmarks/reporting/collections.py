"""Retained query records, graded summaries, and explicitly labeled training exports."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from benchmarks.datasets.photos import save_json
from benchmarks.metrics.statistics import distribution


def candidate_rows(record: dict, query: dict, images: dict) -> list[dict]:
    labels = {r["image_sha256"]: r for r in query["judgments"]}
    result = []
    for rank, candidate in enumerate(record["trace"]["candidates"], 1):
        key = candidate["image_id"]
        image = images[key]
        label = labels.get(key)
        result.append(
            {
                "feature_schema": 1,
                "query_id": query["id"],
                "query": query["text"],
                "language": query["language"],
                "query_family": query["family"],
                "query_split": query["split"],
                "image_id": key,
                "image_path": image["path"],
                "image_split": image["split"],
                "source_document": image["source_document"],
                "duplicate_group": image["duplicate_group"],
                "category": image["category"],
                "candidate_rank": rank,
                "grade": label["grade"] if label is not None else None,
                "label": int(label["grade"] > 0) if label is not None else None,
                "judgment_reason": label["reason"] if label is not None else None,
                "training_eligible": label is not None and query["split"] == image["split"],
                "features": {k: v for k, v in candidate.items() if k != "image_id"},
            }
        )
    return result


def export(output: Path, db: sqlite3.Connection, corpus: dict, bundle: dict, identity: dict):
    records = [
        json.loads(row[0]) for row in db.execute("SELECT payload FROM queries ORDER BY query_id")
    ]
    queries = {q["id"]: q for q in bundle["queries"]}
    images = {r["sha256"]: r for r in corpus["images"]}
    groups: dict[str, list] = defaultdict(list)
    counts: Counter[str] = Counter()
    # Derived exports are regenerated atomically; SQLite is the resumable source of truth.
    handles = {}
    names = [
        "queries.jsonl",
        "candidates.jsonl",
        *[f"training-{s}.jsonl" for s in ("train", "validation", "test")],
    ]
    try:
        for name in names:
            handles[name] = (output / (name + ".tmp")).open("w", encoding="utf-8")
        for record in records:
            handles["queries.jsonl"].write(json.dumps(record, ensure_ascii=False) + "\n")
            query = queries[record["query_id"]]
            if record["status"] != "ok":
                continue
            for row in candidate_rows(record, query, images):
                handles["candidates.jsonl"].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["candidates"] += 1
                if row["training_eligible"]:
                    handles[f"training-{query['split']}.jsonl"].write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    counts[query["split"] + ("_positive" if row["label"] else "_negative")] += 1
            for mode, metrics in record["metrics"].items():
                for tag in (
                    "all",
                    "split:" + query["split"],
                    "language:" + query["language"],
                    "kind:" + query.get("kind", "mixed"),
                    "family:" + query["family"],
                    "exposure:" + query.get("source_exposure", "unspecified"),
                ):
                    groups[mode + "/" + tag].append(metrics)
        for handle in handles.values():
            handle.close()
        for name in names:
            (output / (name + ".tmp")).replace(output / name)
    finally:
        for handle in handles.values():
            handle.close()
    aggregate = {
        group: {key: sum(r[key] for r in values) / len(values) for key in values[0]}
        | {"queries": len(values)}
        for group, values in sorted(groups.items())
    }
    ok = [r for r in records if r["status"] == "ok"]
    summary: dict = {
        "identity": identity,
        "expected_queries": len(queries),
        "processed_queries": len(records),
        "successful_queries": len(ok),
        "failed_queries": len(records) - len(ok),
        "execution_success_rate": len(ok) / len(queries),
        "aggregate": aggregate,
        "exports": dict(counts),
        "hybrid_end_to_end_seconds": distribution([r["seconds"] for r in records]),
        "peak_tree_rss_mib": distribution([r["peak_tree_rss_mib"] for r in records]),
        "candidate_pool_recall": distribution([r["pool_recall"] for r in ok]),
        "limitations": [
            "Relevance uses reviewed known positives; judgments are incomplete.",
            "Unjudged items contribute zero gain to known-judgment metrics, "
            "never a negative training label.",
            "Candidate Recall@50 measures each ranked top 50; "
            "pool recall separately measures the full union.",
            "Full mixed-corpus retrieval; training exports exclude unjudged and cross-split pairs.",
            "Latency includes source checks, artifact verification, "
            "process/model startup, tracing, and thumbnails.",
            "Channel rankings share one hybrid request; no independent channel latency is claimed.",
            "Memory samples coordinator and children; "
            "shared pages can count twice and short spikes can be missed.",
        ],
    }
    with (output / "attempts.jsonl.tmp").open("w", encoding="utf-8") as handle:
        for query_id, attempt, payload in db.execute(
            "SELECT query_id,attempt,payload FROM attempts ORDER BY query_id,attempt"
        ):
            handle.write(
                json.dumps(
                    {"query_id": query_id, "attempt": attempt, "result": json.loads(payload)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    (output / "attempts.jsonl.tmp").replace(output / "attempts.jsonl")
    save_json(output / "summary.json", summary)
    lines = [
        "# Saved-collection retrieval evaluation",
        "",
        f"Images: {identity['snapshot']['unique_images']}; paths: {identity['snapshot']['paths']}.",
        f"Queries: {len(ok)}/{len(queries)} successful; {len(records) - len(ok)} errors.",
        "",
        "| Mode | Split | Queries | Recall@50 | MRR@10 | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode in ("visual", "lexical", "semantic", "hybrid"):
        for split in ("all", "split:train", "split:validation", "split:test"):
            group_metrics = aggregate.get(mode + "/" + split)
            if group_metrics:
                lines.append(
                    f"| {mode} | {split} | {group_metrics['queries']} | "
                    f"{group_metrics['candidate_recall_at_50']:.4f} | "
                    f"{group_metrics['mrr_at_10']:.4f} | {group_metrics['ndcg_at_10']:.4f} |"
                )
    lines.extend(["", "## Interpretation", "", *["- " + item for item in summary["limitations"]]])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
