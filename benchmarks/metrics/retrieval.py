"""Retrieval quality metrics for supplied benchmark judgments."""

import math
from typing import Any


def metrics(ranking: list[str], relevant: list[str]) -> dict[str, Any]:
    """Binary relevance; omitted judgments are not treated as confirmed negatives."""
    positives = set(relevant)
    if not positives:
        return {}
    top = ranking[:10]
    hits = [1 if image_id in positives else 0 for image_id in top]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(positives))))
    return {
        "recall_at_5": len(set(ranking[:5]) & positives) / len(positives),
        "hit_rate_at_5": float(bool(set(ranking[:5]) & positives)),
        "mrr_at_10": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
        "ndcg_at_10": sum(hit / math.log2(i + 2) for i, hit in enumerate(hits)) / ideal,
        "candidate_recall_at_50": len(set(ranking[:50]) & positives) / len(positives),
    }
