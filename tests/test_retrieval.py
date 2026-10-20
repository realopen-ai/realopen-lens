import numpy as np
import pytest

from benchmarks.metrics.retrieval import metrics
from realopen_lens.retrieval.ranking import fuse, lexical_index, rank_lexical, rank_vectors
from realopen_lens.retrieval.candidates import CandidateDepths
from realopen_lens.retrieval.ranking import hybrid_rank


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


@pytest.mark.parametrize("value", [0, -1, 1001, True, 50.5, "50"])
def test_candidate_budgets_reject_unbounded_or_noninteger_depths(value):
    with pytest.raises(ValueError, match="candidate depth"):
        CandidateDepths(semantic=value)


def test_channel_depths_recover_excluded_candidates_without_changing_default_fusion():
    ids = [f"{i:03}" for i in range(300)]
    visual = np.linspace(0.5, 0.1, 300, dtype=np.float32).reshape(-1, 1)
    # Disjoint visual, semantic and keyword leaders exercise all three budgets.
    passages = list(reversed(ids[100:200]))
    semantic = np.linspace(0.99, 0.85, 100, dtype=np.float32).reshape(-1, 1)
    keyword = list(reversed(ids[200:300]))
    args = (visual, np.ones(1), ids, semantic, np.ones(1), passages, "ERR_42", [])
    common = {"lexical_candidates": keyword, "lexical_statistics": (300, {})}
    baseline = {}
    original = hybrid_rank(*args, trace=baseline, **common)
    explicit = {}
    assert (
        hybrid_rank(*args, trace=explicit, candidate_depths=CandidateDepths(), **common) == original
    )
    assert explicit == baseline
    base_pool = {row["image_id"] for row in baseline["candidates"]}
    assert len(base_pool) == 150
    for channel, target in (("visual", "075"), ("semantic", "125"), ("lexical", "225")):
        assert target not in base_pool
        trace = {}
        depths = CandidateDepths(**{channel: 100})
        hybrid_rank(*args, trace=trace, candidate_depths=depths, **common)
        pool = {row["image_id"] for row in trace["candidates"]}
        assert base_pool < pool and target in pool and len(pool) == 200
        assert len(trace["rankings"][channel]) == 100
        assert len(trace["rankings"]["hybrid"]) == 50
        assert trace["candidate_depths"] == depths.as_dict()
