"""Graded known-judgment metrics; unknown candidates are never training negatives."""

import math


def evaluate(ranking: list[str], labels: dict[str, int]) -> dict:
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking repeats an image")
    positives = {key for key, grade in labels.items() if grade > 0}
    if not positives:
        raise ValueError("Quality metrics require a reviewed positive")
    gains = [(2 ** labels.get(key, 0) - 1) for key in ranking[:10]]
    ideal = sorted((2**grade - 1 for grade in labels.values()), reverse=True)[:10]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    return {
        "candidate_recall_at_50": len(set(ranking[:50]) & positives) / len(positives),
        "mrr_at_10": next(
            (1 / i for i, key in enumerate(ranking[:10], 1) if key in positives), 0.0
        ),
        "ndcg_at_10": dcg / idcg,
        "judged_fraction_at_10": (
            sum(key in labels for key in ranking[:10]) / min(10, len(ranking)) if ranking else 0.0
        ),
        "known_positives": len(positives),
    }
