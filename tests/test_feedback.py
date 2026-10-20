"""Persistent explicit judgments, audit history and conservative training exports."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from typer.testing import CliRunner

from realopen_lens.cli import app
from realopen_lens.learning import feedback
from realopen_lens.storage.feedback import FeedbackStore


@pytest.fixture
def saved(tmp_path):
    payload = {
        "version": 1,
        "collection_id": "collection",
        "collection": "personal",
        "query": {"kind": "text", "text": "فاتورة Café"},
        "generation": "old-generation",
        "manifest_sha256": "manifest",
        "pipeline": "pipeline",
        "encoder_spec": {},
        "ranking": {"mode": "hybrid", "siglip_precision": "fp32"},
        "ranking_sources": {"ranking.py": "hash"},
        "results": [
            {
                "image_id": key,
                "rank": rank,
                "visual_similarity": 0.8,
                "width": 100,
                "height": 80,
                "status": "ready",
                "paths": ["private.png"],
                "text": "private OCR",
            }
            for rank, key in enumerate(["one", "two"], 1)
        ],
        "candidates": {"one": {"image_id": "one", "visual_cosine": 0.8}},
    }
    with FeedbackStore(tmp_path) as store:
        run_id = store.add_run(payload)
    return tmp_path, run_id, payload


def test_corrections_removals_and_idempotence_are_auditable(saved):
    root, run_id, payload = saved
    first = feedback.judge(run_id, "one", "relevant", index_dir=root)
    assert feedback.judge(run_id, "one", "relevant", index_dir=root) == first
    feedback.judge(run_id, "one", "irrelevant", index_dir=root)
    feedback.remove_judgment(run_id, "one", index_dir=root)
    feedback.remove_judgment(run_id, "one", index_dir=root)
    assert feedback.saved_search(run_id, index_dir=root)["judgments"] == []
    last = feedback.judge(run_id, "one", "relevant", index_dir=root)
    assert last["revision"] == 4
    events = feedback.judgment_history(run_id, index_dir=root)
    assert [e["label"] for e in events] == ["relevant", "irrelevant", None, "relevant"]
    assert [e["revision"] for e in events] == [1, 2, 3, 4]
    assert feedback.saved_runs(index_dir=root)[0]["query"] == payload["query"]


def test_feedback_rejects_invented_labels_and_results_without_partial_writes(saved):
    root, run_id, _ = saved
    for run, image, label in [
        (run_id, "one", "clicked"),
        (run_id, "one", None),
        (run_id, "not-returned", "relevant"),
        ("missing", "one", "relevant"),
    ]:
        with pytest.raises(ValueError):
            feedback.judge(run, image, label, index_dir=root)
    assert feedback.judgment_history(run_id, index_dir=root) == []
    assert feedback.judge(run_id, "one", "irrelevant", index_dir=root)["revision"] == 1


def test_export_uses_only_latest_active_explicit_labels_and_omits_paths(saved, tmp_path):
    root, run_id, payload = saved
    path = tmp_path / "exports" / "labels.jsonl"
    assert feedback.export_feedback(path, index_dir=root) == 0
    feedback.judge(run_id, "one", "relevant", index_dir=root)
    feedback.judge(run_id, "one", "irrelevant", index_dir=root)
    with pytest.raises(ValueError, match="already exists"):
        feedback.export_feedback(path, index_dir=root)
    assert feedback.export_feedback(path, index_dir=root, overwrite=True) == 1
    text = path.read_text(encoding="utf-8")
    row = json.loads(text)
    assert row["label"] == 0 and row["revision"] == 2
    assert row["query"] == payload["query"] and row["generation"] == "old-generation"
    assert row["candidate"] == payload["candidates"]["one"]
    assert "private" not in text and "split" not in row
    feedback.remove_judgment(run_id, "one", index_dir=root)
    assert feedback.export_feedback(path, index_dir=root, overwrite=True) == 0
    assert path.read_bytes() == b""
    assert not list(path.parent.glob("*.tmp-*"))
    with pytest.raises(ValueError, match="index database"):
        feedback.export_feedback(root / "feedback.sqlite3", index_dir=root, overwrite=True)


def test_export_failure_preserves_previous_file_and_cleans_temporary(saved, tmp_path, monkeypatch):
    root, run_id, _ = saved
    feedback.judge(run_id, "one", "relevant", index_dir=root)
    path = tmp_path / "export.jsonl"
    path.write_text("previous", encoding="utf-8")

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(feedback.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        feedback.export_feedback(path, index_dir=root, overwrite=True)
    assert path.read_text() == "previous" and not list(tmp_path.glob("*.tmp-*"))


def test_concurrent_judgments_are_serialized(saved):
    root, run_id, _ = saved

    def judge(index):
        feedback.judge(run_id, "one", "relevant" if index % 2 else "irrelevant", index_dir=root)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(judge, range(12)))
    history = feedback.judgment_history(run_id, index_dir=root)
    assert [r["revision"] for r in history] == list(range(1, len(history) + 1))
    current = feedback.saved_search(run_id, index_dir=root)["judgments"][0]
    assert current["label"] == history[-1]["label"] and current["revision"] == len(history)


@pytest.mark.parametrize("unrelated", [False, True])
def test_feedback_refuses_unsupported_or_unrelated_databases(tmp_path, unrelated):
    db = sqlite3.connect(tmp_path / "feedback.sqlite3")
    if unrelated:
        db.execute("CREATE TABLE unrelated(value)")
    else:
        db.execute("PRAGMA user_version=99")
    db.close()
    with pytest.raises(ValueError):
        FeedbackStore(tmp_path)


def test_listing_without_feedback_is_readonly_and_validates_limits(tmp_path):
    assert feedback.saved_runs(index_dir=tmp_path) == []
    assert not (tmp_path / "feedback.sqlite3").exists()
    with pytest.raises(ValueError):
        feedback.saved_runs(index_dir=tmp_path, limit=0)
    with pytest.raises(ValueError, match="save-run"):
        feedback.judge("missing", "one", "relevant", index_dir=tmp_path)
    assert not (tmp_path / "feedback.sqlite3").exists()


def test_feedback_cli_supports_set_show_remove_history_and_export(saved, tmp_path):
    root, run_id, _ = saved
    runner = CliRunner()

    def invoke(*args):
        result = runner.invoke(app, ["feedback", *args, "--index-dir", str(root)])
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)

    assert invoke("runs")[0]["run_id"] == run_id
    assert invoke("set", run_id, "one", "relevant")["label"] == "relevant"
    assert invoke("show", run_id)["judgments"][0]["label"] == "relevant"
    assert invoke("export", str(tmp_path / "cli.jsonl")) == 1
    invoke("remove", run_id, "one")
    assert invoke("history", run_id)[-1]["label"] is None
    result = runner.invoke(
        app, ["feedback", "set", run_id, "not-returned", "relevant", "--index-dir", str(root)]
    )
    assert result.exit_code == 1 and "Error:" in result.output and "Traceback" not in result.output


def test_default_export_does_not_clobber_concurrently_created_destination(
    saved, tmp_path, monkeypatch
):
    root, run_id, _ = saved
    path = tmp_path / "concurrent.jsonl"
    feedback.judge(run_id, "one", "relevant", index_dir=root)
    original = feedback.os.link

    def race(source, destination):
        path.write_text("another export", encoding="utf-8")
        return original(source, destination)

    monkeypatch.setattr(feedback.os, "link", race)
    with pytest.raises(ValueError, match="already exists"):
        feedback.export_feedback(path, index_dir=root)
    assert path.read_text(encoding="utf-8") == "another export"
    assert not list(tmp_path.glob("*.tmp-*"))
