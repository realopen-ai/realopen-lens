"""Isolated experiments preserve source labels and refuse historical outputs."""

import json

import pytest

from test_feedback_training import reviewed as reviewed_fixture, app as collection_fixture
from benchmarks.experiments.reviewed_feedback import run, main
from realopen_lens.storage.feedback import FeedbackStore

reviewed = reviewed_fixture
app = collection_fixture


def test_experiment_audits_snapshot_without_changing_source_or_training(reviewed, tmp_path):
    source, review, plan = reviewed
    with FeedbackStore(source, readonly=True) as db:
        before = "\n".join(db.db.iterdump())
    output = tmp_path / "audit"
    progress = []
    result = run(source, output, "personal", None, on_progress=progress.append)
    assert result == {"status": "audited", "runs": 3, "cohorts": 1}
    assert not (output / "index/rerankers").exists()
    assert [event["phase"] for event in progress] == ["snapshotting", "auditing"]
    with FeedbackStore(source, readonly=True) as db:
        assert "\n".join(db.db.iterdump()) == before
    with FeedbackStore(output / "index", readonly=True) as db:
        assert db.db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 6


def test_experiment_executes_existing_training_without_activation(reviewed, tmp_path):
    source, review, plan = reviewed
    output = tmp_path / "trained"
    result = run(source, output, "personal", review)
    assert result["training"]["status"] == "completed"
    assert result["training"]["evaluation"]["evaluated"] == 2
    assert not (output / "index/rerankers/state.json").exists()
    assert (
        json.loads((output / "result.json").read_text(encoding="utf-8"))["prepared"]["pairs"] == 6
    )
    assert not (source / "rerankers").exists()


def test_existing_experiment_is_not_overwritten_on_cli_error(tmp_path, monkeypatch):
    output = tmp_path / "history"
    output.mkdir()
    marker = output / "error.json"
    marker.write_text("original evidence", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["reviewed_feedback", "--source-index", str(tmp_path), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="new experiment"):
        main()
    assert marker.read_text(encoding="utf-8") == "original evidence"
