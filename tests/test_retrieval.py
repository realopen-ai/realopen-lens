import numpy as np
import pytest

from benchmarks.retrieval_metrics import metrics
from realopen_lens.retrieval.ranking import fuse, lexical_index, rank_lexical, rank_vectors


def test_recall_is_not_hit_rate():
    result = metrics(["a", "x", "b", "y", "z"], ["a", "b", "c", "d"])
    assert result["recall_at_5"] == 0.5
    assert result["hit_rate_at_5"] == 1
    assert result["mrr_at_10"] == 1
    assert 0 < result["ndcg_at_10"] < 1
    assert metrics(["x"], ["a"])["mrr_at_10"] == 0
    assert metrics(["x"], []) == {}


def test_passage_ranking_aggregates_images_before_limiting():
    vectors = np.array([[0.9, 0], [0.95, 0], [0.7, 0]], dtype=np.float32)
    assert rank_vectors(vectors, np.array([1, 0]), ["a", "a", "b"], limit=2) == ["a", "b"]


def test_negative_vector_scores_do_not_disappear():
    assert rank_vectors(np.array([[-0.5], [-0.9]]), np.array([1]), ["a", "b"]) == ["a", "b"]


def test_fts_handles_code_and_search_syntax_as_literals():
    database = lexical_index(
        [
            {"image_id": "a", "text": "ModuleNotFoundError: psycopg connection refused"},
            {"image_id": "b", "text": "ملاحظات الاجتماع"},
        ]
    )
    try:
        assert rank_lexical(database, '"ModuleNotFoundError" - connection:') == ["a"]
        assert rank_lexical(database, "الاجتماع") == ["b"]
        assert rank_lexical(database, '" * : ()') == []
    finally:
        database.close()


def test_fusion_counts_each_image_once_per_channel():
    assert fuse([["a", "a", "b"], ["b", "c"]])[0] == "b"
    assert fuse([[], []]) == []


def test_ndcg_discounts_later_relevant_results():
    assert metrics(["a", "b"], ["a", "b"])["ndcg_at_10"] == pytest.approx(1)
    assert metrics(["x", "a", "b"], ["a", "b"])["ndcg_at_10"] < 1
