"""Persistent subsets, pre-retrieval filtering, revisions and scoped provenance."""

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from test_collections import app as collection_fixture, image
from realopen_lens.organization import collections as virtual
from realopen_lens.storage.organization import OrganizationStore, CollectionConflict
from realopen_lens.retrieval import search, images
from realopen_lens.learning import feedback
from realopen_lens.models.manager import digest

app = collection_fixture


@pytest.fixture
def library(app, monkeypatch):
    root, store, calls, run = app
    image(root / "red.png")
    image(root / "green.png", "green")
    run()
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(
        search,
        "query_vector",
        lambda text, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0],
    )
    return root, store, calls, run


def test_virtual_membership_is_idempotent_persistent_and_does_not_touch_sources(library):
    root, store, calls, run = library
    before = {p.name: digest(p) for p in root.iterdir()}
    original_calls = list(calls)
    group = virtual.create(store, "personal", "  الفواتير  ")
    key = digest(root / "red.png")
    group = virtual.change_members(store, "personal", group["id"], [key, key], group["revision"])
    revision = group["revision"]
    assert (
        virtual.change_members(store, "personal", group["id"], [key], revision)["revision"]
        == revision
    )
    result = virtual.browse(store, "personal", group["id"])
    assert [r["image_id"] for r in result["results"]] == [key]
    assert result["scope"]["name"] == "الفواتير"
    assert result["member_count"] == 1 and result["unavailable_count"] == 0
    group = virtual.rename(store, "personal", group["id"], "Receipts", revision)
    assert virtual.list_collections(store, "personal")[0]["name"] == "Receipts"
    assert before == {p.name: digest(p) for p in root.iterdir()}
    assert calls == original_calls
    with pytest.raises(CollectionConflict):
        virtual.change_members(store, "personal", group["id"], [key], revision, remove=True)
    virtual.remove(store, "personal", group["id"], group["revision"])
    with OrganizationStore(store, readonly=True) as db:
        assert db.db.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 0
    assert virtual.list_collections(store, "personal") == []


def test_names_conflicts_and_invalid_members_do_not_partially_apply(library):
    root, store, calls, run = library
    group = virtual.create(store, "personal", "Invoices")
    with pytest.raises(ValueError, match="already exists"):
        virtual.create(store, "personal", "INVOICES")
    for name in ("", " ", "x" * 81, "a\x00b"):
        with pytest.raises(ValueError):
            virtual.rename(store, "personal", group["id"], name, group["revision"])
    with pytest.raises(ValueError, match="available images"):
        virtual.change_members(
            store, "personal", group["id"], [digest(root / "red.png"), "0" * 64], 1
        )
    assert virtual.browse(store, "personal", group["id"])["member_count"] == 0
    run(collection="other")
    with pytest.raises(ValueError, match="source collection"):
        virtual.browse(store, "other", group["id"])
    with pytest.raises(CollectionConflict):
        virtual.browse(store, "personal", group["id"], revision=999)


def test_unavailable_members_survive_refresh_and_reappear_with_same_content(library):
    root, store, calls, run = library
    path = root / "red.png"
    original = path.read_bytes()
    key = digest(path)
    group = virtual.create(store, "personal", "Documents")
    group = virtual.change_members(store, "personal", group["id"], [key], 1)
    generation = virtual.browse(store, "personal", group["id"])["generation"]
    path.unlink()
    result = virtual.browse(store, "personal", group["id"])
    assert result["results"] == [] and result["unavailable_count"] == 1
    run()
    assert virtual.list_collections(store, "personal")[0]["unavailable_count"] == 1
    with pytest.raises(CollectionConflict, match="Source index"):
        virtual.browse(store, "personal", group["id"], generation=generation)
    path.write_bytes(original)
    run()
    assert virtual.browse(store, "personal", group["id"])["total"] == 1
    assert virtual.list_collections(store, "personal")[0]["revision"] == group["revision"]


@pytest.mark.parametrize("mode", ["hybrid", "visual", "semantic", "lexical"])
def test_search_filters_before_candidate_generation_even_below_full_library_top50(
    app, monkeypatch, mode
):
    root, store, calls, run = app
    for i in range(60):
        image(root / f"{i}.png", (i + 1, 0, 0))
    run()
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(
        search,
        "query_vector",
        lambda text, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0],
    )
    all_ids = sorted(digest(path) for path in root.iterdir())
    target = all_ids[-1]
    baseline = search.search_collection("red ERR_42", index_dir=store, mode=mode, limit=50)
    assert target not in {r.image_id for r in baseline}
    group = virtual.create(store, "personal", "Only the last candidate")
    group = virtual.change_members(store, "personal", group["id"], [target], 1)
    scoped = search.search_collection(
        "red ERR_42",
        index_dir=store,
        mode=mode,
        virtual_collection=group["id"],
        virtual_revision=group["revision"],
    )
    assert [r.image_id for r in scoped] == [target]


def test_similar_scopes_results_but_can_reuse_query_outside_group(library):
    root, store, calls, run = library
    key, other = digest(root / "red.png"), digest(root / "green.png")
    group = virtual.create(store, "personal", "Green")
    group = virtual.change_members(store, "personal", group["id"], [other], 1)
    trace = {}
    found = images.search_similar(
        image_id=key, index_dir=store, virtual_collection=group["id"], trace=trace
    )
    assert [r.image_id for r in found] == [other]
    assert trace["reused_embedding"] and trace["scope"]["revision"] == group["revision"]


def test_membership_changes_during_search_fail_and_do_not_record_feedback(library, monkeypatch):
    root, store, calls, run = library
    key = digest(root / "red.png")
    group = virtual.create(store, "personal", "Docs")
    group = virtual.change_members(store, "personal", group["id"], [key], 1)

    def progress(value):
        if value["phase"] == "encoding_visual_query":
            virtual.change_members(
                store, "personal", group["id"], [key], group["revision"], remove=True
            )

    with pytest.raises(CollectionConflict):
        feedback.search_and_record(
            "red", index_dir=store, virtual_collection=group["id"], on_progress=progress
        )
    assert feedback.saved_runs(index_dir=store) == []


def test_scoped_feedback_preserves_scope_in_export(library, tmp_path):
    root, store, calls, run = library
    key = digest(root / "red.png")
    group = virtual.create(store, "personal", "Red docs")
    virtual.change_members(store, "personal", group["id"], [key], 1)
    saved = feedback.search_and_record("red", index_dir=store, virtual_collection=group["id"])
    feedback.judge(saved.run_id, key, "relevant", index_dir=store)
    path = tmp_path / "labels.jsonl"
    feedback.export_feedback(path, index_dir=store)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["scope"]["id"] == group["id"] and row["scope"]["member_count"] == 1


def test_store_rejects_unrelated_database(tmp_path):
    with sqlite3.connect(tmp_path / "organization.sqlite3") as db:
        db.execute("CREATE TABLE unrelated(id INTEGER)")
    with pytest.raises(ValueError, match="unrelated"):
        OrganizationStore(tmp_path)


def test_cli_group_lifecycle_and_scoped_search(library):
    from typer.testing import CliRunner
    from realopen_lens.cli import app as cli

    root, store, calls, run = library
    runner = CliRunner()

    def invoke(*args):
        response = runner.invoke(cli, [*args, "--index-dir", str(store)])
        assert response.exit_code == 0, response.output
        return json.loads(response.stdout)

    group = invoke("groups", "create", "Work documents")
    key = digest(root / "red.png")
    group = invoke("groups", "add", group["id"], key, "--revision", "1")
    results = invoke("search", "red", "--virtual-collection", group["id"])
    assert [r["image_id"] for r in results] == [key]
    assert invoke("groups", "images", group["id"])["total"] == 1
    invoke("groups", "delete", group["id"], "--revision", str(group["revision"]))
    assert invoke("groups", "list") == []


def test_scoped_feedback_cannot_enter_unscoped_training_review(library, tmp_path):
    from realopen_lens.learning.dataset import prepare

    root, store, calls, run = library
    key = digest(root / "red.png")
    group = virtual.create(store, "personal", "Docs")
    virtual.change_members(store, "personal", group["id"], [key], 1)
    saved = feedback.search_and_record("red", index_dir=store, virtual_collection=group["id"])
    feedback.judge(saved.run_id, key, "relevant", index_dir=store)
    path = tmp_path / "labels.jsonl"
    feedback.export_feedback(path, index_dir=store)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "version": 1,
                "reviewer": "fixture",
                "feedback_file": path.name,
                "feedback_sha256": digest(path),
                "runs": {saved.run_id: {"family": "red", "split": "train"}},
                "image_groups": {key: key},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scope-aware"):
        prepare(store, review, tmp_path / "dataset")


def test_virtual_search_uses_baseline_until_scope_aware_model_evaluation_exists(
    library, monkeypatch
):
    from realopen_lens.learning import registry

    root, store, calls, run = library
    key = digest(root / "red.png")
    group = virtual.create(store, "personal", "Red")
    virtual.change_members(store, "personal", group["id"], [key], 1)

    def unmanaged(*args):
        raise AssertionError(
            "Managed model must not be applied to an unvalidated scoped distribution"
        )

    monkeypatch.setattr(registry, "active", unmanaged)
    found = search.search_collection("red", index_dir=store, virtual_collection=group["id"])
    assert [r.image_id for r in found] == [key]
    with pytest.raises(ValueError, match="scoped evaluation"):
        search.search_collection(
            "red", index_dir=store, virtual_collection=group["id"], reranker=Path("model.json")
        )
    with pytest.raises(ValueError, match="index database"):
        feedback.export_feedback(store / "organization.sqlite3", index_dir=store, overwrite=True)


def test_source_rename_preserves_membership_after_index_refresh(library):
    root, store, calls, run = library
    key = digest(root / "red.png")
    group = virtual.create(store, "personal", "Saved")
    group = virtual.change_members(store, "personal", group["id"], [key], 1)
    (root / "red.png").rename(root / "renamed.png")
    assert virtual.browse(store, "personal", group["id"])["unavailable_count"] == 1
    run()
    result = virtual.browse(store, "personal", group["id"])
    assert result["total"] == 1
    assert result["results"][0]["paths"][0].endswith("renamed.png")
    assert result["scope"]["revision"] == group["revision"]


def test_concurrent_first_writes_preserve_all_virtual_collections(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    ready = Barrier(6)

    def create(number):
        ready.wait(timeout=10)
        with OrganizationStore(tmp_path) as db:
            return db.create("parent", f"Group {number}")["id"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(create, range(6)))
    assert len(set(ids)) == 6
    with OrganizationStore(tmp_path, readonly=True) as db:
        assert len(db.list_collections("parent")) == 6
