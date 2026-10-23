"""Global retrieval preserves scope, provenance, bounded encoders and ranking."""

from pathlib import Path

import numpy as np
import pytest

from test_collections import app as _app, image
from test_local_api import web as _web, collection_fixture as _collection_fixture, completed
from realopen_lens.retrieval.library import LibrarySession, LIBRARY_ID
from realopen_lens.retrieval.search import search_collection
from realopen_lens.runtime.resident import ResidentQueryEncoder
from realopen_lens.learning.feedback import change_judgment, export_feedback, saved_search
from realopen_lens.storage.catalog import Catalog

app = _app
web = _web
collection_fixture = _collection_fixture


@pytest.fixture
def encoders(monkeypatch):
    calls = []
    monkeypatch.setattr(ResidentQueryEncoder, "verify", lambda *a: None)

    def encode(self, query, kind, models, threads, precision):
        calls.append((kind, precision))
        vector = np.zeros(768 if kind == "siglip2" else 384, dtype=np.float32)
        vector[0 if "red" in query else 1] = 1
        return vector

    monkeypatch.setattr(ResidentQueryEncoder, "encode", encode)
    return calls


def populate(app, tmp_path):
    root, store, calls, run = app
    image(root / "one.png")
    image(root / "green.png", "green")
    run()
    second = tmp_path / "second"
    second.mkdir()
    image(second / "same.png")
    image(second / "blue.png", "blue")
    run(root=str(second), collection="second", siglip_precision="fp32")
    return root, store, second, run


def test_mixed_library_encodes_once_per_precision_and_preserves_locations(app, tmp_path, encoders):
    root, store, second, run = populate(app, tmp_path)
    session = LibrarySession(index_dir=store)
    try:
        profile, trace = {}, {}
        result = session.search("red", mode="visual", trace=trace, profile=profile)
        assert len(result["results"]) == 3
        first = result["results"][0]
        assert set(first["paths"]) == {str(root / "one.png"), str(second / "same.png")}
        assert {s["collection"] for s in first["sources"]} == {"personal", "second"}
        assert len(encoders) == 2 and set(encoders) == {("siglip2", "int8"), ("siglip2", "fp32")}
        assert profile["visual_encodings"] == 2 and profile["semantic_encodings"] == 0
        assert trace["searchable_images"] == 3 and len(trace["sources"]) == 2
        encoders.clear()
        session.search("red", mode="hybrid")
        assert sum(k == "e5" for k, p in encoders) == 1
        assert sum(k == "siglip2" for k, p in encoders) == 2
    finally:
        session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.search("red")


@pytest.mark.parametrize("mode", ["hybrid", "visual", "semantic", "lexical"])
def test_one_folder_library_matches_existing_ranking(app, encoders, monkeypatch, mode):
    root, store, calls, run = app
    for name, color in [("one", "red"), ("two", "green"), ("three", "blue")]:
        image(root / (name + ".png"), color)
    run()
    monkeypatch.setattr("realopen_lens.retrieval.search.verify", lambda *a: None)
    monkeypatch.setattr(
        "realopen_lens.retrieval.search.query_vector",
        lambda q, k, *args: ResidentQueryEncoder().encode(q, k, None, 2, "int8"),
    )
    expected = search_collection("red ERR_42", index_dir=store, mode=mode, limit=50)
    session = LibrarySession(index_dir=store)
    try:
        observed = session.search("red ERR_42", mode=mode)["results"]
    finally:
        session.close()
    assert [r["image_id"] for r in observed] == [r.image_id for r in expected]
    assert [r["visual_similarity"] for r in observed] == [r.visual_similarity for r in expected]


def test_library_feedback_does_not_leak_into_folder_search(app, tmp_path, encoders, monkeypatch):
    root, store, second, run = populate(app, tmp_path)
    session = LibrarySession(index_dir=store)
    try:
        result = session.search("red ERR_42", save_run=True)
        key = result["results"][0]["image_id"]
        change_judgment(result["run_id"], key, "irrelevant", index_dir=store)
        again = session.search("red ERR_42")
        assert key not in {r["image_id"] for r in again["results"]}
        assert key in {
            r["image_id"] for r in session.search("red ERR_42", use_feedback=False)["results"]
        }
        hidden = session.search("red ERR_42", mode="lexical", hidden_only=True)
        assert hidden["total"] == 1 and len(hidden["results"][0]["sources"]) == 2
        run = saved_search(result["run_id"], index_dir=store)
        assert run["collection_id"] == LIBRARY_ID and len(run["library_sources"]) == 2
        assert export_feedback(tmp_path / "feedback.jsonl", index_dir=store) == 1
        monkeypatch.setattr("realopen_lens.retrieval.search.verify", lambda *a: None)
        monkeypatch.setattr(
            "realopen_lens.retrieval.search.query_vector",
            lambda q, k, *a: ResidentQueryEncoder().encode(q, k, None, 2, "int8"),
        )
        folder = search_collection("red ERR_42", index_dir=store)
        assert key in {r.image_id for r in folder}
        change_judgment(result["run_id"], key, None, index_dir=store)
        assert session.search("red ERR_42", mode="lexical", hidden_only=True)["total"] == 0
    finally:
        session.close()


def test_unavailable_sources_are_reported_and_never_returned(app, tmp_path, encoders):
    root, store, second, run = populate(app, tmp_path)
    (root / "one.png").unlink()
    (root / "green.png").write_bytes(b"changed")
    session = LibrarySession(index_dir=store)
    try:
        result = session.search("red", mode="visual")
        assert result["unavailable"] == [{"collection": "personal", "images": 2}]
        assert len(result["results"]) == 2
        assert all(s["collection"] == "second" for r in result["results"] for s in r["sources"])
    finally:
        session.close()


def test_library_progress_cancellation_closes_owned_session(app, encoders):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    session = LibrarySession(index_dir=store)
    phases = []

    def progress(value):
        phases.append(value["phase"])
        if value["phase"] == "library_encoding":
            raise KeyboardInterrupt()

    try:
        with pytest.raises(KeyboardInterrupt):
            session.search("red", on_progress=progress)
        assert "library_source_validation" in phases
    finally:
        session.close()
    assert session.encoder is None


def test_library_lexical_search_needs_no_models(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    run()

    def forbidden(*args):
        raise AssertionError("Lexical search loaded a model")

    monkeypatch.setattr(ResidentQueryEncoder, "verify", forbidden)
    session = LibrarySession(index_dir=store, models=Path("absent"))
    try:
        assert session.search("ERR_42", mode="lexical")["results"]
    finally:
        session.close()


def test_pending_empty_and_limit_errors_are_explicit(app, tmp_path, encoders, monkeypatch):
    root, store, calls, run = app
    catalog = Catalog(store)
    catalog.ensure_collection("pending", root)
    catalog.close()
    session = LibrarySession(index_dir=store)
    try:
        assert session.search("red")["pending_folders"] == ["pending"]
        with pytest.raises(ValueError):
            session.search(" ")
        with pytest.raises(ValueError):
            session.search("red", limit=101)
        image(root / "one.png")
        run()
        monkeypatch.setattr("realopen_lens.retrieval.library.MAX_IMAGES", 0)
        with pytest.raises(ValueError, match="100,000"):
            session.search("red")
    finally:
        session.close()


def test_library_api_uses_existing_jobs_and_sse(web, encoders):
    client, root, store, run = web
    response = client.post(
        "/api/tasks/search", json={"query": "red", "all_folders": True, "save_run": True}
    )
    assert response.status_code == 202
    task = completed(client, response.json()["id"])
    assert task["status"] == "completed", task
    assert task["collection"] == "All folders" and task["all_folders"]
    assert task["result"]["results"][0]["sources"]
    assert "event: progress" in client.get(f"/api/tasks/{task['id']}/events").text
    result = task["result"]["results"][0]
    label = client.put(
        f"/api/feedback/runs/{task['result']['run_id']}/{result['image_id']}",
        json={"label": "irrelevant"},
    )
    assert label.status_code == 200, label.text
    hidden = client.get("/api/library/hidden", params={"query": "red"})
    assert hidden.status_code == 200 and hidden.json()["total"] == 1
    assert (
        client.post(
            "/api/tasks/search",
            json={"query": "red", "all_folders": True, "virtual_collection": "a" * 32},
        ).status_code
        == 422
    )


def test_cross_format_pixel_duplicates_keep_all_locations_and_memory(app, tmp_path, encoders):
    from realopen_lens.organization.duplicates import scan

    root, store, calls, run = app
    image(root / "one.png")
    run()
    second = tmp_path / "other"
    second.mkdir()
    image(second / "copy.bmp")
    run(root=str(second), collection="other", siglip_precision="fp32")
    scan(store, "personal")
    scan(store, "other")
    session = LibrarySession(index_dir=store)
    try:
        result = session.search("red", save_run=True)
        assert len(result["results"]) == 1
        first = result["results"][0]
        assert len(first["duplicate_ids"]) == 1 and len(first["sources"]) == 2
        change_judgment(result["run_id"], first["image_id"], "irrelevant", index_dir=store)
        assert session.search("red")["results"] == []
        assert session.search("red", mode="lexical", hidden_only=True)["total"] == 1
    finally:
        session.close()


def test_corrupt_folder_fails_library_explicitly(app, encoders):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    catalog = Catalog(store, readonly=True)
    try:
        _, gen = catalog.snapshot("personal")
    finally:
        catalog.close()
    (store / gen["manifest"]).write_text("{}", encoding="utf-8")
    session = LibrarySession(index_dir=store)
    try:
        with pytest.raises(ValueError, match="corrupt"):
            session.search("red")
    finally:
        session.close()


def test_library_training_requires_separate_evaluation_protocol(app, encoders, tmp_path):
    from realopen_lens.learning.dataset import prepare
    import json
    from realopen_lens.models.manager import digest

    # Library runs have multiple source specs and must never masquerade as
    # existing per-collection 50/channel training sessions.
    root, store, calls, run = app
    image(root / "one.png")
    run()
    session = LibrarySession(index_dir=store)
    try:
        saved = session.search("red", save_run=True)
    finally:
        session.close()
    change_judgment(saved["run_id"], saved["results"][0]["image_id"], "relevant", index_dir=store)
    # Exercise the actual preparation API using its explicit review manifest.
    assert "library_sources" in saved_search(saved["run_id"], index_dir=store)
    exported = tmp_path / "feedback.jsonl"
    export_feedback(exported, index_dir=store)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "version": 1,
                "reviewer": "test",
                "feedback_file": exported.name,
                "feedback_sha256": digest(exported),
                "runs": {saved["run_id"]: {"split": "train", "family": "red"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="library-aware"):
        prepare(store, review, tmp_path / "dataset")


def test_partitioning_does_not_give_folders_ranking_votes(app, tmp_path, encoders):
    from benchmarks.experiments.library_search import partition

    root, store, calls, run = app
    # More than a single result budget; tied encoder outputs still have a
    # deterministic global ordering independent of which folder holds them.
    for number in range(65):
        image(root / f"{number}.png", (number + 1, 0, 0))
    run()
    baseline = LibrarySession(index_dir=store)
    try:
        original = baseline.search("red ERR_42", use_feedback=False)["results"]
    finally:
        baseline.close()
    divided = tmp_path / "divided"
    partition([store] * 4, "personal", divided)
    library = LibrarySession(index_dir=divided)
    try:
        partitioned = library.search("red ERR_42", use_feedback=False)["results"]
    finally:
        library.close()
    assert [r["image_id"] for r in original] == [r["image_id"] for r in partitioned]
    assert [r["visual_similarity"] for r in original] == [
        r["visual_similarity"] for r in partitioned
    ]


def test_cli_all_folders_preserves_folder_command_and_rejects_scope_options(app, encoders):
    import json
    from typer.testing import CliRunner
    from realopen_lens.cli import app as cli

    root, store, calls, run = app
    image(root / "one.png")
    run()
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "red", "--all-folders", "--index-dir", str(store)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["results"][0]["sources"][0]["collection"] == "personal"
    conflict = runner.invoke(
        cli, ["search", "red", "--all-folders", "--collection", "other", "--index-dir", str(store)]
    )
    assert conflict.exit_code == 1 and "folder-specific" in conflict.output


def test_library_media_survives_folder_rename_and_refuses_unregistered_source(web, encoders):
    client, root, store, run = web
    task = client.post("/api/tasks/search", json={"query": "red", "all_folders": True}).json()
    result = completed(client, task["id"])["result"]["results"][0]
    source = result["sources"][0]
    url = f"/api/sources/{source['collection_id']}/images/{source['image_id']}/thumbnail"
    params = {"generation": source["generation"]}
    assert client.get(url, params=params).status_code == 200
    renamed = client.post(
        "/api/collections/rename",
        json={
            "collection": "personal",
            "collection_id": source["collection_id"],
            "name": "renamed",
        },
    )
    assert renamed.status_code == 200, renamed.text
    assert client.get(url, params=params).status_code == 200
    removed = client.post(
        "/api/collections/remove",
        json={"collection": "renamed", "collection_id": source["collection_id"]},
    )
    assert removed.status_code == 200
    assert client.get(url, params=params).status_code == 404


def test_library_judgment_changes_after_only_a_new_format_alias_remains(app, tmp_path, encoders):
    from realopen_lens.organization.duplicates import scan

    root, store, calls, run = app
    image(root / "original.png")
    image(root / "keep.png", "green")
    run()
    second = tmp_path / "other"
    second.mkdir()
    image(second / "copy.bmp")
    run(root=str(second), collection="other", siglip_precision="fp32")
    scan(store, "personal")
    scan(store, "other")
    session = LibrarySession(index_dir=store)
    try:
        first = session.search("red", save_run=True)
        original = first["results"][0]
        decision = change_judgment(
            first["run_id"], original["image_id"], "relevant", index_dir=store
        )["memory"]
        # Remove whichever physical source owns the canonical old result ID.
        from realopen_lens.models.manager import digest

        if digest(root / "original.png") == original["image_id"]:
            (root / "original.png").unlink()
            run()
            scan(store, "personal")
        else:
            (second / "copy.bmp").unlink()
            run(root=str(second), collection="other", siglip_precision="fp32")
            scan(store, "other")
        next_run = session.search("red", save_run=True)
        replacement = next_run["results"][0]
        assert replacement["image_id"] != original["image_id"]
        assert replacement["feedback"]["event_id"] == decision["event_id"]
        changed = change_judgment(
            next_run["run_id"],
            replacement["image_id"],
            "irrelevant",
            index_dir=store,
            expected_event_id=decision["event_id"],
        )
        assert changed["memory"]["label"] == "irrelevant"
        assert all(
            r["image_id"] != replacement["image_id"] for r in session.search("red")["results"]
        )
    finally:
        session.close()


def test_only_cross_precision_visual_ranking_is_marked_experimental(app, tmp_path, encoders):
    root, store, second, run = populate(app, tmp_path)
    session = LibrarySession(index_dir=store)
    try:
        for mode in ("hybrid", "visual", "semantic", "lexical"):
            result = session.search("red", mode=mode)
            assert result["ranking"]["mixed_precision_experimental"] == (
                mode in {"hybrid", "visual"}
            )
    finally:
        session.close()


def test_library_session_serializes_concurrent_queries(app, encoders, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    root, store, calls, run = app
    image(root / "one.png")
    run()
    entered = threading.Event()
    release = threading.Event()
    second = threading.Event()
    original = ResidentQueryEncoder.encode
    active = 0
    maximum = 0

    def encoding(*args):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        entered.set()
        assert release.wait(5)
        try:
            return original(*args)
        finally:
            active -= 1

    monkeypatch.setattr(ResidentQueryEncoder, "encode", encoding)
    session = LibrarySession(index_dir=store)

    def another():
        second.set()
        return session.search("red", mode="visual")

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(session.search, "red", mode="visual")
            assert entered.wait(5)
            other = workers.submit(another)
            assert second.wait(5)
            release.set()
            assert first.result()["results"] == other.result()["results"]
        assert maximum == 1
    finally:
        release.set()
        session.close()


def test_library_similarity_reuses_each_precision_and_excludes_query(app, tmp_path, encoders):
    root, store, second, run = populate(app, tmp_path)
    from realopen_lens.models.manager import digest

    key = digest(root / "one.png")
    session = LibrarySession(index_dir=store)
    try:
        trace = {}
        result = session.similar(image_id=key, trace=trace, save_run=True)
        assert len(result["results"]) == 2
        assert all(key not in {r["image_id"], *r["duplicate_ids"]} for r in result["results"])
        assert trace["reused_image_precisions"] == ["fp32", "int8"]
        assert result["ranking"]["mixed_precision_experimental"]
        assert not result["feedback"]["enabled"]
        assert encoders == []
        saved = saved_search(result["run_id"], index_dir=store)
        assert saved["query"] == {"kind": "image", "image_id": key}
        assert len(saved["library_sources"]) == 2
        assert len(session.similar(root / "one.png", include_self=True)["results"]) == 3
        with pytest.raises(ValueError, match="not in the source"):
            session.similar(image_id="0" * 64)
    finally:
        session.close()


def test_library_similarity_encodes_only_missing_precision(app, tmp_path, monkeypatch):
    from realopen_lens.models.manager import digest
    from realopen_lens.retrieval import images

    root, store, second, run = populate(app, tmp_path)
    image(root / "unique.png", "yellow")
    run()
    calls = []
    monkeypatch.setattr(images, "verify_image_encoder", lambda *a: None)

    def encode(path, models, threads, precision):
        calls.append(precision)
        vector = np.zeros(768, dtype=np.float32)
        vector[0] = 1
        return vector

    monkeypatch.setattr(images, "image_vector", encode)
    session = LibrarySession(index_dir=store)
    try:
        result = session.similar(image_id=digest(root / "unique.png"))
        assert calls == ["fp32"]
        assert len(result["results"]) == 3
        calls.clear()
        external = tmp_path / "external.png"
        image(external, "purple")
        session.similar(external)
        assert sorted(calls) == ["fp32", "int8"]

        def changed(path, *args):
            image(path, "orange")
            return encode(path, *args)

        monkeypatch.setattr(images, "image_vector", changed)
        with pytest.raises(ValueError, match="Query image changed"):
            session.similar(external)
    finally:
        session.close()


def test_library_similarity_api_and_cli(web, encoders):
    from typer.testing import CliRunner
    from realopen_lens.cli import app as cli

    client, root, store, run = web
    folder = "personal"
    key = client.get(f"/api/collections/{folder}/images").json()["results"][0]["image_id"]
    response = client.post(
        "/api/tasks/similar",
        json={
            "collection": folder,
            "image_id": key,
            "all_folders": True,
            "save_run": True,
        },
    )
    assert response.status_code == 202
    task = completed(client, response.json()["id"])
    assert task["status"] == "completed", task
    assert task["all_folders"]
    assert all(r["image_id"] != key for r in task["result"]["results"])
    assert (
        client.post(
            "/api/tasks/similar",
            json={
                "collection": folder,
                "image_id": key,
                "all_folders": True,
                "virtual_collection": "a" * 32,
            },
        ).status_code
        == 422
    )
    uploaded = client.post(
        f"/api/tasks/upload?collection={folder}&all_folders=true",
        content=(root / "red.png").read_bytes(),
    )
    assert uploaded.status_code == 202
    upload_task = completed(client, uploaded.json()["id"])
    assert upload_task["status"] == "completed", upload_task
    assert upload_task["all_folders"]
    result = CliRunner().invoke(
        cli,
        [
            "similar",
            "--image-id",
            key,
            "--collection",
            folder,
            "--all-folders",
            "--index-dir",
            str(store),
        ],
    )
    assert result.exit_code == 0, result.output


def test_library_similarity_matches_single_folder_without_models(app):
    from realopen_lens.retrieval.images import search_similar
    from realopen_lens.models.manager import digest

    root, store, calls, run = app
    image(root / "one.png")
    image(root / "two.png", "blue")
    image(root / "three.png", "green")
    run()
    key = digest(root / "one.png")
    expected = search_similar(image_id=key, index_dir=store)
    session = LibrarySession(index_dir=store)
    try:
        actual = session.similar(image_id=key)
        assert [r["image_id"] for r in actual["results"]] == [r.image_id for r in expected]
        assert [r["visual_similarity"] for r in actual["results"]] == pytest.approx(
            [r.visual_similarity for r in expected]
        )
        assert not actual["ranking"]["mixed_precision_experimental"]
    finally:
        session.close()


def test_library_similarity_excludes_cross_format_pixel_copies(app, tmp_path, monkeypatch):
    from realopen_lens.organization.duplicates import scan
    from realopen_lens.retrieval import images

    root, store, calls, run = app
    image(root / "one.png")
    run()
    second = tmp_path / "other"
    second.mkdir()
    image(second / "copy.bmp")
    run(root=str(second), collection="other")
    scan(store, "personal")
    scan(store, "other")
    session = LibrarySession(index_dir=store)
    try:
        assert session.similar(root / "one.png")["results"] == []
        external = tmp_path / "external.tiff"
        image(external)
        monkeypatch.setattr(images, "verify_image_encoder", lambda *a: None)
        monkeypatch.setattr(images, "image_vector", lambda *a: np.eye(1, 768)[0])
        assert session.similar(external)["results"] == []
        recorded = session.similar(external, include_self=True, save_run=True)
        assert len(recorded["results"]) == 1
        change_judgment(
            recorded["run_id"], recorded["results"][0]["image_id"], "irrelevant", index_dir=store
        )
        # Image judgments are retained for export, not exact-text query memory.
        assert len(session.similar(external, include_self=True)["results"]) == 1

        def cancel(event):
            if event["phase"] == "library_encoding":
                raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            session.similar(external, on_progress=cancel)
    finally:
        session.close()
