"""Candidate-depth comparisons must preserve paired evidence and reject mixed runs."""

import json

import pytest

from benchmarks.datasets.photos import save_json
from benchmarks.metrics.collection import evaluate
from benchmarks.metrics.statistics import distribution
from benchmarks.reporting.candidates import compare


def fixture_run(path, depth, pool, ranking):
    path.mkdir()
    quality = evaluate(ranking, {"target": 2})
    record = {
        "query_id": "q-ar",
        "status": "ok",
        "query": {"language": "ar", "judgments": [{"image_sha256": "target", "grade": 2}]},
        "trace": {"candidates": [{"image_id": key} for key in pool]},
        "pool_recall": float("target" in pool),
        "metrics": {"hybrid": quality},
        "seconds": 1,
        "peak_tree_rss_mib": 100,
    }
    save_json(
        path / "summary.json",
        {
            "identity": {
                "snapshot": "fixed",
                "candidate_depths": {
                    "visual": depth,
                    "semantic": depth,
                    "lexical": depth,
                },
            },
            "failed_queries": 0,
            "processed_queries": 1,
            "expected_queries": 1,
            "quality_deferred_queries": 0,
            "aggregate": {"hybrid/all": quality},
            "candidate_pool_recall": distribution([record["pool_recall"]]),
            "candidate_pool_recall_by_group": {},
            "candidate_pool_size": distribution([len(pool)]),
            "hybrid_end_to_end_seconds": distribution([1]),
            "peak_tree_rss_mib": distribution([100]),
        },
    )
    (path / "queries.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_pool_recovery_is_reported_separately_from_unchanged_final_ranking(tmp_path):
    base, wider = tmp_path / "base", tmp_path / "wider"
    fixture_run(base, 50, ["noise"], ["noise"])
    fixture_run(wider, 200, ["noise", "target"], ["noise"])
    report = compare([base, wider], tmp_path / "report")
    paired = report["paired_queries"][-1]
    assert paired["new_pool_positives"] == ["target"]
    assert paired["pool_recall"] == 1
    assert paired["hybrid_delta"]["candidate_recall_at_50"] == 0
    assert paired["hybrid_metrics"]["mrr_at_10"] == 0
    summary = json.loads((wider / "summary.json").read_text(encoding="utf-8"))
    summary["identity"]["snapshot"] = "changed"
    save_json(wider / "summary.json", summary)
    with pytest.raises(ValueError, match="differ beyond"):
        compare([base, wider], tmp_path / "bad")


@pytest.mark.parametrize(
    "field", ["failed_queries", "quality_deferred_queries", "expected_queries"]
)
def test_candidate_comparison_rejects_incomplete_or_unscored_runs(tmp_path, field):
    run = tmp_path / "run"
    fixture_run(run, 50, ["target"], ["target"])
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary[field] += 1
    save_json(run / "summary.json", summary)
    with pytest.raises(ValueError, match="complete, successful, scored"):
        compare([run], tmp_path / "bad")
