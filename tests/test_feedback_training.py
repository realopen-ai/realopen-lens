"""Reviewed-data isolation, production replay and explicit model lifecycle."""

import copy
import json

import numpy as np
import pytest

from test_collections import app as collection_fixture, image
from realopen_lens.learning import dataset, feedback, registry, workflow
from realopen_lens.retrieval import search
from realopen_lens.runtime.resident import ResidentQueryEncoder
from realopen_lens.models.manager import digest

app = collection_fixture


@pytest.fixture(params=["int8", "fp32"])
def reviewed(app, monkeypatch, tmp_path, request):
    root, store, calls, run = app
    for number in range(6):
        # Different bytes identify independent synthetic sources.
        image(root / f"{number}.png", (number * 30 + 10, number * 20, 0))
    run(siglip_precision=request.param)

    def vector(text, kind, *args):
        return np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0]

    monkeypatch.setattr(search, "query_vector", vector)
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(ResidentQueryEncoder, "verify", lambda *a: None)
    monkeypatch.setattr(ResidentQueryEncoder, "encode", lambda self, *args: vector(*args))
    plan = {
        "version": 1,
        "reviewer": "synthetic regression fixture; not real relevance",
        "independent_holdout": False,
        "runs": {},
        "image_groups": {},
    }
    used = set()
    for split in ("train", "validation", "test"):
        record = feedback.search_and_record(f"red {split}", index_dir=store, limit=50)
        ids = [r.image_id for r in record.results if r.image_id not in used][:2]
        assert len(ids) == 2
        for label, key in enumerate(ids):
            feedback.judge(
                record.run_id, key, "relevant" if label else "irrelevant", index_dir=store
            )
            plan["image_groups"][key] = key
        used.update(ids)
        plan["runs"][record.run_id] = {"family": split, "split": split}
    export = tmp_path / "feedback.jsonl"
    feedback.export_feedback(export, index_dir=store)
    plan.update(feedback_file=export.name, feedback_sha256=digest(export))
    review = tmp_path / "review.json"
    review.write_text(json.dumps(plan), encoding="utf-8")
    return store, review, plan


def test_reviewed_training_evaluates_before_explicit_activation_and_rollback(reviewed, tmp_path):
    store, review, plan = reviewed
    directory = tmp_path / "dataset"
    frozen = dataset.prepare(store, review, directory)
    assert frozen["pairs"] == 6 and not frozen["independent_holdout"]
    progress = []
    result = workflow.train(directory / "dataset.json", store, on_progress=progress.append)
    assert result["status"] == "completed"
    assert result["evaluation"]["queries"] == 2
    assert registry.state(store)["active"] == {}
    phases = {p["phase"] for p in progress}
    assert {"training", "selecting", "evaluating", "retrieval"} <= phases
    artifacts = store / "rerankers" / result["id"]
    assert len((artifacts / "queries.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ValueError, match="acknowledgment"):
        registry.change(store, "personal", result["id"])
    registry.change(store, "personal", result["id"], acknowledge_experimental=True)
    assert registry.active(store, "personal") == artifacts / "model.json"
    trace = {}
    search.search_collection("red", index_dir=store, trace=trace)
    assert trace["reranker_sha256"] == result["model_sha256"]
    registry.change(store, "personal", None)
    assert registry.active(store, "personal") is None
    registry.change(store, "personal", result["id"], acknowledge_experimental=True, rollback=True)
    assert registry.state(store)["history"][-1]["operation"] == "rollback"
    (artifacts / "model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="intact artifact"):
        registry.active(store, "personal")


@pytest.mark.parametrize("problem", ["family", "group", "query", "stale", "checksum", "precision"])
def test_review_rejects_leakage_and_stale_or_incompatible_inputs(reviewed, tmp_path, problem):
    store, review, original = reviewed
    plan = copy.deepcopy(original)
    runs = list(plan["runs"])
    if problem == "family":
        plan["runs"][runs[1]]["family"] = plan["runs"][runs[0]]["family"]
    elif problem == "group":
        keys = list(plan["image_groups"])
        plan["image_groups"][keys[2]] = plan["image_groups"][keys[0]]
    elif problem == "query":
        with feedback.FeedbackStore(store) as db:
            payload = db.run(runs[1])
            payload["query"] = db.run(runs[0])["query"]
            db.db.execute(
                "UPDATE searches SET payload=? WHERE id=?", (json.dumps(payload), runs[1])
            )
            db.db.commit()
    elif problem == "stale":
        saved = feedback.saved_search(runs[0], index_dir=store)
        feedback.remove_judgment(runs[0], saved["judgments"][0]["image_id"], index_dir=store)
    elif problem == "checksum":
        plan["feedback_sha256"] = "0" * 64
    elif problem == "precision":
        with feedback.FeedbackStore(store) as db:
            payload = db.run(runs[0])
            payload["ranking"]["siglip_precision"] = (
                "fp32" if payload["ranking"]["siglip_precision"] == "int8" else "int8"
            )
            db.db.execute(
                "UPDATE searches SET payload=? WHERE id=?", (json.dumps(payload), runs[0])
            )
            db.db.commit()
    review.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        dataset.prepare(store, review, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_cancellation_preserves_interrupted_run_and_never_activates(reviewed, tmp_path):
    store, review, plan = reviewed
    dataset.prepare(store, review, tmp_path / "frozen")

    def cancel(progress):
        if progress["phase"] == "training":
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        workflow.train(tmp_path / "frozen/dataset.json", store, on_progress=cancel)
    assert registry.versions(store)[0]["status"] == "interrupted"
    assert registry.state(store)["active"] == {}


def test_production_errors_are_retained_and_block_activation(reviewed, tmp_path, monkeypatch):
    store, review, plan = reviewed
    dataset.prepare(store, review, tmp_path / "frozen")
    path = tmp_path / "frozen/dataset.json"
    from realopen_lens.retrieval.session import SearchSession

    original = SearchSession.search

    def changed(self, query, **options):
        results = original(self, query, **options)
        if query == "red test":
            options["trace"]["generation"] = "missing-generation"
        return results

    monkeypatch.setattr(SearchSession, "search", changed)
    result = workflow.train(path, store)
    assert result["status"] == "completed_with_errors"
    assert result["evaluation"]["failed"] == 1
    assert result["evaluation"]["by_split"]["test"]["evaluated"] == 0
    records = (store / "rerankers" / result["id"] / "queries.jsonl").read_text(encoding="utf-8")
    assert len(records.splitlines()) == 2
    assert "generation changed" in records
    with pytest.raises(ValueError, match="completed evaluation"):
        registry.change(store, "personal", result["id"], acknowledge_experimental=True)


def test_modified_frozen_groups_are_rejected_before_training(reviewed, tmp_path):
    store, review, plan = reviewed
    dataset.prepare(store, review, tmp_path / "frozen")
    path = tmp_path / "frozen/dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    a, b = data["sessions"][:2]
    key = next(iter(b["image_groups"]))
    b["image_groups"][key] = next(iter(a["image_groups"].values()))
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="group leaks"):
        workflow.train(path, store)
    assert not (store / "rerankers").exists()


def test_restricted_process_inspection_does_not_fail_quality_evaluation(
    reviewed, tmp_path, monkeypatch
):
    import psutil

    store, review, plan = reviewed
    dataset.prepare(store, review, tmp_path / "frozen")

    def denied(*args, **kwargs):
        raise PermissionError("process enumeration restricted")

    monkeypatch.setattr(psutil.Process, "children", denied)
    result = workflow.train(tmp_path / "frozen/dataset.json", store)
    assert result["status"] == "completed"
    assert result["evaluation"]["memory_sampling_incomplete_queries"] == 2


def test_modified_frozen_labels_require_new_review(reviewed, tmp_path):
    store, review, plan = reviewed
    dataset.prepare(store, review, tmp_path / "frozen")
    path = tmp_path / "frozen/dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for session in data["sessions"]:
        if session["split"] == "train":
            for key in session["labels"]:
                session["labels"][key] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot changed"):
        workflow.train(path, store)
