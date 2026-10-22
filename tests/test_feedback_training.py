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


def test_rename_preserves_frozen_and_new_training_and_active_model_rollback(reviewed, tmp_path):
    from realopen_lens.organization.folders import change
    from realopen_lens.storage.catalog import Catalog

    store, review, plan = reviewed
    directory = tmp_path / "before"
    dataset.prepare(store, review, directory)
    first = workflow.train(directory / "dataset.json", store)
    registry.change(store, "personal", first["id"], acknowledge_experimental=True)
    catalog = Catalog(store, readonly=True)
    collection_id = catalog.collection("personal")["id"]
    catalog.close()
    change(store, "personal", collection_id, "Renamed")
    assert registry.active(store, "Renamed") == store / "rerankers" / first["id"] / "model.json"
    # A dataset frozen before rename still resolves its source by ID.
    again = workflow.train(directory / "dataset.json", store)
    assert again["status"] == "completed", again
    # Unchanged historical feedback remains eligible for a newly prepared dataset.
    after = tmp_path / "after"
    assert dataset.prepare(store, review, after)["pairs"] == 6
    registry.change(store, "Renamed", None)
    registry.change(store, "Renamed", first["id"], acknowledge_experimental=True, rollback=True)
    assert registry.active(store, "Renamed") == store / "rerankers" / first["id"] / "model.json"


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


def test_feedback_audit_is_readonly_and_separates_provenance_cohorts(reviewed):
    from realopen_lens.learning.review import audit

    store, review_path, plan = reviewed
    with feedback.FeedbackStore(store, readonly=True) as db:
        before = "\n".join(db.db.iterdump())
    result = audit(store)
    with feedback.FeedbackStore(store, readonly=True) as db:
        assert "\n".join(db.db.iterdump()) == before
    assert len(result["runs"]) == 3 and len(result["cohorts"]) == 1
    assert len(result["cohorts"][0]["eligible_run_ids"]) == 3
    assert result["cohorts"][0]["labels"] == {"relevant": 3, "irrelevant": 3}
    runs = list(plan["runs"])
    with feedback.FeedbackStore(store) as db:
        saved = db.run(runs[1])
        saved["ranking_sources"]["search.py"] = "different-recorded-source"
        db.db.execute("UPDATE searches SET payload=? WHERE id=?", (json.dumps(saved), runs[1]))
        db.db.commit()
    changed = audit(store)
    assert len(changed["cohorts"]) == 2
    assert sum(sum(c["labels"].values()) for c in changed["cohorts"]) == 6


def test_known_duplicate_bridge_blocks_mislabeled_manual_groups(reviewed, monkeypatch, tmp_path):
    from realopen_lens.learning import review as reviewing

    store, review_path, plan = reviewed
    keys = list(plan["image_groups"])
    component = min(keys[0], keys[2])
    constraints = {
        "version": 1,
        "collection_id": "test-source",
        "components": {keys[0]: component, keys[2]: component},
        "evidence": {
            "groups": [{"kind": "confirmed", "image_ids": [keys[0], "unlabeled-bridge", keys[2]]}]
        },
    }
    monkeypatch.setattr(reviewing, "duplicate_constraints", lambda *a: constraints)
    with pytest.raises(ValueError, match="Known exact/confirmed duplicate"):
        dataset.prepare(store, review_path, tmp_path / "leaking")
    assert not (tmp_path / "leaking").exists()


def test_frozen_duplicate_constraints_are_rechecked_at_training(reviewed, tmp_path):
    store, review_path, plan = reviewed
    dataset.prepare(store, review_path, tmp_path / "frozen-groups")
    path = tmp_path / "frozen-groups/dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = list(plan["image_groups"])
    data["duplicate_constraints"]["personal"]["components"] = {keys[0]: keys[0], keys[2]: keys[0]}
    with pytest.raises(ValueError, match="Frozen exact/confirmed duplicate"):
        dataset.validate_frozen(data)


def test_duplicate_review_links_constrain_splits_without_propagating_labels(reviewed):
    from realopen_lens.learning.review import duplicate_constraints
    from realopen_lens.organization import duplicates
    from realopen_lens.storage.artifacts import load_manifest, atomic_bytes
    from realopen_lens.storage.catalog import Catalog

    store, review_path, plan = reviewed
    with_catalog = Catalog(store, readonly=True)
    found, generation = with_catalog.snapshot("personal")
    manifest = load_manifest(with_catalog, generation)
    with_catalog.close()
    keys = list(plan["image_groups"])
    # Synthetic reviewed constraints test closure through an unlabeled identity;
    # this does not assert a real perceptual match or create relevance judgments.
    value = {
        "policy": duplicates.POLICY,
        "key": duplicates._key(manifest),
        "exact": [{"id": "exact", "image_ids": [keys[0], "bridge"]}],
        "near": [
            {"id": "yes", "image_ids": ["bridge", keys[2]]},
            {"id": "no", "image_ids": [keys[2], keys[4]]},
        ],
    }
    atomic_bytes(store / "duplicates" / (value["key"] + ".json"), json.dumps(value).encode())
    atomic_bytes(
        store / "duplicates/reviews" / (found["id"] + ".json"),
        json.dumps({"yes": "confirmed", "no": "dismissed"}).encode(),
    )
    result = duplicate_constraints(store, "personal")
    assert result["components"][keys[0]] == result["components"][keys[2]]
    assert keys[4] not in result["components"]
    assert len(result["evidence"]["groups"]) == 2
    with feedback.FeedbackStore(store, readonly=True) as db:
        assert db.db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 6


def test_audit_and_prepare_keep_duplicate_presentation_policies_separate(reviewed, tmp_path):
    from realopen_lens.learning.review import audit

    store, review_path, plan = reviewed
    run_id = next(iter(plan["runs"]))
    with feedback.FeedbackStore(store) as db:
        saved = db.run(run_id)
        saved["ranking"].update(duplicate_policy="pixels-v1", duplicate_snapshot="scan-key")
        db.db.execute("UPDATE searches SET payload=? WHERE id=?", (json.dumps(saved), run_id))
        db.db.commit()
    report = audit(store)
    assert len(report["cohorts"]) == 2
    export = review_path.with_name(plan["feedback_file"])
    feedback.export_feedback(export, index_dir=store, overwrite=True)
    plan["feedback_sha256"] = digest(export)
    review_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="Mixed duplicate presentation"):
        dataset.prepare(store, review_path, tmp_path / "mixed-policy")


def test_training_audit_cli_reports_cohorts_without_training(reviewed):
    from typer.testing import CliRunner
    from realopen_lens.cli import app

    store, review_path, plan = reviewed
    result = CliRunner().invoke(app, ["training", "audit", "--index-dir", str(store)])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["runs"]) == 3
    assert not (store / "rerankers").exists()


def test_production_pair_order_and_progress_ignore_training_sessions(reviewed, tmp_path):
    from realopen_lens.learning.evaluation import production

    store, review_path, plan = reviewed
    directory = tmp_path / "ordered"
    dataset.prepare(store, review_path, directory)
    trained = workflow.train(directory / "dataset.json", store)
    data = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    by_split = {session["split"]: session for session in data["sessions"]}
    data["sessions"] = [by_split["validation"], by_split["train"], by_split["test"]]
    model = store / "rerankers" / trained["id"] / "model.json"
    progress = []
    output = tmp_path / "paired.jsonl"
    production(data, model, store, tmp_path / "unused-models", output, progress.append)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["execution_order"] for r in records] == ["hybrid_first", "logistic_first"]
    phases = [event for event in progress if event["phase"] == "evaluating"]
    assert [(p["query"], p["queries"]) for p in phases] == [(1, 2), (2, 2)]


def test_personalized_searches_and_superseded_labels_cannot_train(reviewed, tmp_path):
    from realopen_lens.learning.review import audit

    store, review, plan = reviewed
    first = next(iter(plan["runs"]))
    with feedback.FeedbackStore(store) as db:
        payload = db.run(first)
        ids = [row["image_id"] for row in db.judgments(first)]
        payload["ranking"]["feedback_memory"] = {"enabled": True, "applied": True}
        personalized = db.add_run(payload)
        db.judge(personalized, ids[0], "relevant")
    # Current query memory must not contaminate baseline evaluation/training.
    with pytest.raises(ValueError, match="superseded"):
        dataset.prepare(store, review, tmp_path / "invalid")
    reasons = [reason for row in audit(store)["runs"] for reason in row["reasons"]]
    assert any("query memory applied" in reason for reason in reasons)
    assert any("superseded" in reason for reason in reasons)
    # A fresh export still preserves the contradicted history explicitly.
    output = tmp_path / "fresh.jsonl"
    feedback.export_feedback(output, index_dir=store)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert any(row.get("superseded") for row in rows if row["run_id"] == first)
