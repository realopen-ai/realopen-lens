"""Focused regression coverage for experimental reranking."""

from benchmarks.datasets.photos import save_json
from realopen_lens.learning.features import NAMES
from realopen_lens.learning.logistic import load


def candidate(index):
    return {
        "image_id": f"{index:03}",
        "visual_cosine": 1 - index / 100,
        "semantic_cosine": None,
        "lexical_evidence": 0.0,
        "semantic_strength": 0.0,
        "hybrid_score": 1 - index / 100,
        "lexical_bm25": None,
        "channel_ranks": {
            "visual": index + 1 if index < 50 else None,
            "lexical": None,
            "semantic": None,
        },
        "ocr_characters": 0,
    }


def test_hybrid50_preserves_membership_and_tail_while_union_can_demote(tmp_path):
    artifact = {
        "version": 2,
        "kind": "logistic_regression",
        "feature_schema": 1,
        "candidate_budget": 50,
        "features": list(NAMES),
        "mean": [0] * len(NAMES),
        "scale": [1] * len(NAMES),
        "weights": [-1] + [0] * (len(NAMES) - 1),
        "intercept": 0,
        "compatibility": {"model_registry": {}, "inference_sha256": "fixture"},
        "feature_profile": "full",
        "rerank_scope": "hybrid50",
    }
    path = tmp_path / "model.json"
    save_json(path, artifact)
    rows = [candidate(i) for i in range(70)]
    model = load(path)
    assert model.rank(list(reversed(rows))) == [
        row["image_id"] for row in rows[:50][::-1] + rows[50:]
    ]
    assert set(model.rank(rows)[:50]) == {row["image_id"] for row in rows[:50]}
    artifact["rerank_scope"] = "union"
    save_json(path, artifact)
    assert load(path).rank(rows).index("000") == 69
    artifact["version"] = 1
    del artifact["rerank_scope"]
    del artifact["feature_profile"]
    save_json(path, artifact)
    assert load(path).scope == "union" and load(path).rank(rows).index("000") == 69
