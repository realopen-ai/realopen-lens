"""Category-level quality and resource summary statistics."""

from collections import Counter, defaultdict
import numpy as np


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        **{
            name: float(np.percentile(values, p))
            for name, p in [("median", 50), ("p95", 95), ("p99", 99)]
        },
        "max": float(max(values)),
    }


def summarize(records: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for field in ["all", "category", "subcategory", "format", "split", "difficulty", "resolution"]:
        grouped: dict[str, list] = defaultdict(list)
        for item in records:
            labels = (
                ["all"]
                if field == "all"
                else item.get(field, []) if field == "difficulty" else [item.get(field, "unknown")]
            )
            for label in labels:
                grouped[str(label)].append(item)
        result[field] = {}
        for label, items in sorted(grouped.items()):
            counts = Counter(r["status"] for r in items)
            evaluated = [r for r in items if r["quality_status"] != "not_evaluated"]
            quality_items = [row["quality"] for row in items if row.get("quality")]
            totals = {
                key: sum(row[key] for row in quality_items)
                for key in (
                    "character_errors",
                    "reference_characters",
                    "word_errors",
                    "reference_words",
                    "matched_tokens",
                    "reference_tokens",
                    "observed_tokens",
                    "matched_key_phrases",
                    "reference_key_phrases",
                )
            }
            quality_metrics = {
                name: totals[num] / totals[den] if totals[den] else None
                for name, num, den in (
                    ("cer", "character_errors", "reference_characters"),
                    ("wer", "word_errors", "reference_words"),
                    ("token_recall", "matched_tokens", "reference_tokens"),
                    ("token_precision", "matched_tokens", "observed_tokens"),
                    ("key_phrase_recall", "matched_key_phrases", "reference_key_phrases"),
                )
            }
            result[field][label] = {
                "ocr": {"evaluated_images": len(quality_items), **totals, **quality_metrics},
                "count": len(items),
                "statuses": dict(counts),
                "execution_success_rate": sum(
                    r["status"] not in ("error", "timeout") for r in items
                )
                / len(items),
                "quality_evaluated": len(evaluated),
                "quality_pass_rate": (
                    sum(r["quality_status"] == "pass" for r in evaluated) / len(evaluated)
                    if evaluated
                    else None
                ),
                "latency_seconds": distribution([r["latency_seconds"] for r in items]),
                "peak_tree_rss_mib": distribution(
                    [r["peak_tree_rss_mib"] for r in items if r["peak_tree_rss_mib"] is not None]
                ),
                "timeout_rate": counts["timeout"] / len(items),
            }
    return result
