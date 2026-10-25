"""Local API contracts, browser isolation and observable background jobs."""

import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from test_collections import app as _collection_fixture, image
from realopen_lens.api import tasks as jobs
from realopen_lens.indexing.service import IndexResult
from realopen_lens.models.manager import digest
from realopen_lens.runtime.resident import ResidentQueryEncoder

try:
    from fastapi.testclient import TestClient
    from realopen_lens.api.app import create_app
except ModuleNotFoundError as error:
    if error.name not in {"fastapi", "starlette", "pydantic"}:
        raise
    pytest.skip("Install the web extra to test the local API", allow_module_level=True)

collection_fixture = _collection_fixture


def test_setup_check_returns_actionable_results_with_live_task_progress(web, monkeypatch):
    from realopen_lens.runtime import readiness

    client, root, store, run = web
    monkeypatch.setattr(
        readiness,
        "artifact_path",
        lambda directory, name, key: store / "missing-models" / name / key,
    )
    monkeypatch.setattr(
        readiness,
        "installed_ocr",
        lambda: {
            "executable": "tesseract",
            "version": "tesseract 5",
            "installed_languages": ["eng"],
        },
    )
    response = client.post("/api/tasks/setup", json={"source": str(root), "languages": "eng+ara"})
    assert response.status_code == 202
    task = completed(client, response.json()["id"])
    assert task["status"] == "completed" and task["result"]["ready"] is False
    checks = {check["id"]: check for check in task["result"]["checks"]}
    assert checks["workspace"]["ok"] and checks["source"]["ok"]
    assert not checks["models"]["ok"] and "ara" in checks["ocr"]["detail"]
    assert "event: progress" in client.get(f"/api/tasks/{task['id']}/events").text
    assert client.get("/api/collections/personal/images").json()["total"] == 2
    assert client.post("/api/tasks/setup", json={"source": str(root.parent)}).status_code == 403
    assert client.post("/api/tasks/setup", json={"languages": "eng+--help"}).status_code == 422


def test_launcher_language_preference_is_live_and_not_tk_bound(tmp_path):
    preference = {"languages": "eng"}
    app = create_app(
        index_dir=tmp_path / "index", preferred_ocr_languages=lambda: preference["languages"]
    )
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/api/session").json()["default_ocr_languages"] == "eng"
        preference["languages"] = "eng+fra"
        assert client.get("/api/session").json()["default_ocr_languages"] == "eng+fra"


def test_setup_verification_can_be_cancelled(web, monkeypatch):
    from realopen_lens.runtime import readiness

    client, root, store, run = web
    started, release = threading.Event(), threading.Event()

    def check(*args, on_progress, **kwargs):
        started.set()
        assert release.wait(5)
        on_progress({"phase": "Verifying local model files"})
        pytest.fail("Cancelled verification should not produce a ready report")

    monkeypatch.setattr(readiness, "inspect", check)
    task = client.post("/api/tasks/setup", json={}).json()
    assert started.wait(5)
    try:
        assert client.post(f"/api/tasks/{task['id']}/cancel").status_code == 200
    finally:
        release.set()
    assert completed(client, task["id"])["status"] == "cancelled"


@pytest.fixture
def web(collection_fixture, monkeypatch):
    root, store, calls, run = collection_fixture
    image(root / "red.png")
    image(root / "green.png", "green")
    run()
    monkeypatch.setattr(ResidentQueryEncoder, "verify", lambda *a: None)
    monkeypatch.setattr(
        ResidentQueryEncoder,
        "encode",
        lambda self, text, kind, *args: np.eye(
            1, 768 if kind == "siglip2" else 384, dtype=np.float32
        )[0],
    )
    app = create_app(index_dir=store, allowed_roots=(root,), port=8765)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/api/session").status_code == 200
        yield client, root, store, run


def completed(client, task_id):
    for _ in range(100):
        state = client.get("/api/tasks/" + task_id).json()
        if state["status"] in jobs.TERMINAL:
            return state
        time.sleep(0.02)
    pytest.fail("Task did not finish")


def test_home_imports_are_default_and_explicit_roots_still_restrict(tmp_path, monkeypatch):
    home = tmp_path / "home"
    photos = home / "Pictures" / "photos"
    photos.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    submitted = []

    def capture(self, kind, request):
        submitted.append((kind, request))
        return {"id": "captured"}

    monkeypatch.setattr(jobs.TaskManager, "submit", capture)
    for overrides, allowed in (
        ({}, True),
        ({"allowed_roots": ()}, False),
        ({"allowed_roots": (outside,)}, False),
    ):
        app = create_app(index_dir=tmp_path / "index", **overrides)
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            session = client.get("/api/session").json()
            if allowed:
                assert session["allowed_roots"] == [str(home.resolve())]
            response = client.post(
                "/api/tasks/index", json={"root": str(photos), "collection": "photos"}
            )
            assert response.status_code == (202 if allowed else 403)
            if allowed:
                assert submitted[-1][1]["root"] == str(photos.resolve())
                response = client.post("/api/tasks/index", json={"root": str(outside)})
                assert response.status_code == 403
    assert len(submitted) == 1


def test_folder_browser_lists_only_directories_with_paging_and_bounds(web, tmp_path):
    client, root, store, run = web
    for number in range(205):
        (root / f"album-{number:03d}").mkdir()
    (root / ".hidden").mkdir()
    first = client.get("/api/folders").json()
    assert first["path"] == str(root.resolve())
    assert first["parent"] is None
    assert len(first["folders"]) == 200
    assert first["folders"][0]["name"] == "album-000"
    assert first["next_offset"] == 200
    second = client.get("/api/folders", params={"path": str(root), "offset": 200}).json()
    assert len(second["folders"]) == 5 and second["next_offset"] is None
    nested = client.get("/api/folders", params={"path": first["folders"][0]["path"]}).json()
    assert nested["parent"] == str(root.resolve()) and nested["folders"] == []
    assert client.get("/api/folders", params={"path": str(tmp_path)}).status_code == 403
    assert client.get("/api/folders", params={"offset": -1}).status_code == 422
    client.cookies.clear()
    assert client.get("/api/folders").status_code == 401


def test_index_name_conflict_is_rejected_before_queuing_and_refresh_is_allowed(web, monkeypatch):
    client, root, store, run = web
    other = root / "photos"
    other.mkdir()
    submitted = []
    monkeypatch.setattr(
        jobs.TaskManager, "submit", lambda self, *args: submitted.append(args) or {"id": "test"}
    )
    response = client.post("/api/tasks/index", json={"root": str(other), "collection": "personal"})
    assert response.status_code == 409
    assert "Use another collection name" in response.json()["detail"]
    assert submitted == []
    response = client.post("/api/tasks/index", json={"root": str(other), "collection": "photos"})
    assert response.status_code == 202
    response = client.post("/api/tasks/index", json={"root": str(root), "collection": "personal"})
    assert response.status_code == 202
    assert len(submitted) == 2


def test_launcher_precision_preference_is_live_and_does_not_change_collections(tmp_path):
    preferred = {"value": "fp32"}
    app = create_app(
        index_dir=tmp_path / "store", preferred_siglip_precision=lambda: preferred["value"]
    )
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/api/session").json()["default_siglip_precision"] == "fp32"
        preferred["value"] = "int8"
        assert client.get("/api/session").json()["default_siglip_precision"] == "int8"
        assert client.get("/api/collections").json() == []


def test_folder_browser_does_not_follow_links_outside_import_roots(web, tmp_path):
    client, root, store, run = web
    outside = tmp_path / "outside-folder"
    outside.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks unavailable on this host")
    assert not client.get("/api/folders").json()["folders"]
    assert client.get("/api/folders", params={"path": str(link)}).status_code == 403


def test_local_session_rejects_foreign_browser_and_host(web):
    client, root, store, run = web
    client.cookies.clear()
    assert client.get("/api/collections").status_code == 401
    assert client.get("/api/session", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/session", headers={"Host": "evil.example"}).status_code == 400
    response = client.get("/api/session")
    assert (
        "HttpOnly" in response.headers["set-cookie"]
        and "SameSite=strict" in response.headers["set-cookie"]
    )
    assert (
        client.post(
            "/api/tasks/search", json={"query": "red"}, headers={"Origin": "null"}
        ).status_code
        == 403
    )
    assert client.get("/api/collections").status_code == 200
    assert "no-store" in response.headers["cache-control"]


def test_browse_and_media_are_bound_to_indexed_content(web):
    client, root, store, run = web
    data = client.get("/api/collections/personal/images?limit=1").json()
    assert data["total"] == 2 and len(data["results"]) == 1
    key = data["results"][0]["image_id"]
    base = "/api/collections/personal/images/" + key
    response = client.get(base + "/thumbnail")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    response = client.get(base + "/original")
    assert response.status_code == 200 and digest(Path(data["results"][0]["paths"][0])) == key
    assert client.get(base + "/original?generation=other-collection").status_code == 404
    assert client.get("/api/collections/personal/images/bad-id/original").status_code == 404
    assert client.get("/api/collections/personal/images?limit=999").status_code == 400
    Path(data["results"][0]["paths"][0]).write_bytes(b"changed")
    assert client.get(base + "/original").status_code == 404
    assert client.get("/api/collections/personal/images").json()["total"] == 1


def test_search_similarity_and_feedback_share_production_operations(web):
    client, root, store, run = web
    response = client.post("/api/tasks/search", json={"query": "ERR_42", "save_run": True})
    assert response.status_code == 202
    state = completed(client, response.json()["id"])
    assert state["status"] == "completed", state
    saved = state["result"]["run_id"]
    key = state["result"]["results"][0]["image_id"]
    assert (
        client.put(f"/api/feedback/runs/{saved}/{key}", json={"label": "relevant"}).status_code
        == 200
    )
    assert client.get(f"/api/feedback/runs/{saved}").json()["judgments"][0]["label"] == "relevant"
    assert client.delete(f"/api/feedback/runs/{saved}/{key}").status_code == 200
    assert client.get(f"/api/feedback/runs/{saved}").json()["judgments"] == []
    assert (
        client.put(
            f"/api/feedback/runs/{saved}/not-returned", json={"label": "relevant"}
        ).status_code
        == 400
    )
    assert (
        client.put(f"/api/feedback/runs/{saved}/{key}", json={"label": "clicked"}).status_code
        == 422
    )
    response = client.post("/api/tasks/similar", json={"image_id": key})
    state = completed(client, response.json()["id"])
    assert state["status"] == "completed" and key not in [
        r["image_id"] for r in state["result"]["results"]
    ]
    assert not any(
        "query" in json.loads(p.read_text()) for p in (store / "api-tasks").glob("*.json")
    )
    assert all(
        "result" not in json.loads(p.read_text()) for p in (store / "api-tasks").glob("*.json")
    )


def test_indexing_is_background_and_sse_reports_live_counts_and_outcome(web, monkeypatch):
    client, root, store, run = web
    release = threading.Event()
    started = threading.Event()

    def index(*args, **kwargs):
        kwargs["on_progress"]({"phase": "ocr", "phase_completed_images": 1, "unique_images": 2})
        started.set()
        assert release.wait(5)
        kwargs["on_progress"]({"phase": "vision", "phase_completed_images": 2, "unique_images": 2})
        return IndexResult(
            "job", "completed_with_errors", "personal", 2, 1, 0, 1, 0, "generation", 1
        )

    monkeypatch.setattr(jobs, "index_collection", index)
    response = client.post("/api/tasks/index", json={"root": str(root)})
    assert response.status_code == 202 and started.wait(1)
    task_id = response.json()["id"]
    # The request finished and other routes remain responsive while the task waits.
    assert client.get("/api/collections").status_code == 200
    assert client.get("/api/tasks/" + task_id).json()["progress"]["phase_completed_images"] == 1
    threading.Timer(0.65, release.set).start()
    response = client.get(f"/api/tasks/{task_id}/events", headers={"Last-Event-ID": "0"})
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert len(frames) >= 2 and frames[0]["status"] == "running"
    assert frames[-1]["status"] == "completed_with_errors"
    assert frames[-1]["result"]["failed_files"] == 1
    # Reconnection receives the latest terminal state, rather than losing completion.
    assert "completed_with_errors" in client.get(f"/api/tasks/{task_id}/events").text


def test_import_boundaries_queue_limit_cancel_and_failed_tasks(web, monkeypatch, tmp_path):
    client, root, store, run = web
    assert client.post("/api/tasks/index", json={"root": str(tmp_path)}).status_code == 403
    entered = threading.Event()
    release = threading.Event()

    def index(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        kwargs["on_progress"]({"phase": "ocr"})
        raise ValueError("local models unavailable")

    monkeypatch.setattr(jobs, "index_collection", index)
    tasks = [client.post("/api/tasks/index", json={"root": str(root)}).json() for _ in range(4)]
    assert entered.wait(1)
    assert client.post("/api/tasks/index", json={"root": str(root)}).status_code == 429
    client.post("/api/tasks/" + tasks[1]["id"] + "/cancel")
    release.set()
    assert completed(client, tasks[0]["id"])["status"] == "failed"
    assert completed(client, tasks[1]["id"])["status"] == "cancelled"
    assert "local models unavailable" in completed(client, tasks[2]["id"])["error"]


def test_uploads_are_bounded_and_temporary(web):
    client, root, store, run = web
    assert client.post("/api/tasks/upload", content=b"").status_code == 400
    key = digest(root / "red.png")
    response = client.post("/api/tasks/upload", content=(root / "red.png").read_bytes())
    assert response.status_code == 202
    state = completed(client, response.json()["id"])
    assert state["status"] == "completed" and key not in [
        r["image_id"] for r in state["result"]["results"]
    ]
    assert not list((store / "api-uploads").iterdir())


def test_task_restart_marks_unfinished_work_interrupted_without_replaying_queries(tmp_path):
    directory = tmp_path / "api-tasks"
    directory.mkdir()
    state = {
        "id": "old",
        "kind": "search",
        "collection": "personal",
        "status": "running",
        "phase": "encoding",
        "revision": 1,
        "created_at": time.time(),
        "progress": {},
        "error": None,
    }
    (directory / "old.json").write_text(json.dumps(state), encoding="utf-8")
    manager = jobs.TaskManager(tmp_path, Path("models"))
    try:
        assert manager.snapshot("old")["status"] == "interrupted"
        assert manager.snapshot("old")["result"] is None
    finally:
        manager.close()


def test_verified_gallery_cache_invalidates_when_artifact_changes(web):
    from realopen_lens.storage.catalog import Catalog
    from realopen_lens.storage.artifacts import load_manifest

    client, root, store, run = web
    first = client.get("/api/collections/personal/images").json()["results"][0]
    route = "/api/collections/personal/images/" + first["image_id"] + "/thumbnail"
    assert client.get(route).status_code == 200
    catalog = Catalog(store, readonly=True)
    try:
        _, generation = catalog.snapshot("personal")
        manifest = load_manifest(catalog, generation)
    finally:
        catalog.close()
    (store / manifest["artifacts"]["vision"]["path"]).write_bytes(b"changed")
    assert client.get(route).status_code == 404


def test_training_review_rejects_unscoped_inputs_and_unavailable_datasets(web):
    client, root, store, run = web
    assert client.get("/api/training/models").json()["active"] == {}
    response = client.post(
        "/api/tasks/prepare-training", json={"review": {"feedback_file": "../feedback.sqlite3"}}
    )
    assert response.status_code == 400
    assert client.post("/api/tasks/train", json={"dataset_id": "a" * 32}).status_code == 400
    assert client.post("/api/tasks/train", json={"dataset_id": "../escape"}).status_code == 422
    assert (
        client.post("/api/training/activation", json={"model_version": "../escape"}).status_code
        == 422
    )
    assert client.post("/api/training/activation", json={}).json()["active"] == {}


def test_training_tasks_use_existing_queue_and_observable_progress(web, monkeypatch):
    client, root, store, run = web
    from realopen_lens.learning import dataset, workflow

    def prepare(index_dir, review, output, on_progress):
        assert index_dir == store.resolve()
        assert review.is_file()
        on_progress({"phase": "preparing", "completed": 1, "total": 3})
        output.mkdir(parents=True)
        (output / "dataset.json").write_text("{}", encoding="utf-8")
        return {"dataset": str(output), "queries": 3}

    def train(path, index_dir, models, on_progress):
        assert path.name == "dataset.json"
        on_progress({"phase": "training", "step": 2, "max_steps": 100})
        return {"id": "b" * 32, "status": "completed"}

    monkeypatch.setattr(dataset, "prepare", prepare)
    monkeypatch.setattr(workflow, "train", train)
    queued = client.post(
        "/api/tasks/prepare-training", json={"review": {"feedback_file": "a" * 32 + ".jsonl"}}
    )
    assert queued.status_code == 202
    prepared = completed(client, queued.json()["id"])
    assert prepared["kind"] == "prepare_training" and prepared["status"] == "completed"
    dataset_id = Path(prepared["result"]["dataset"]).name
    response = client.post("/api/tasks/train", json={"dataset_id": dataset_id})
    trained = completed(client, response.json()["id"])
    assert trained["kind"] == "train" and trained["status"] == "completed"
    assert trained["progress"]["step"] == 2
    events = client.get(f"/api/tasks/{trained['id']}/events")
    assert '"kind": "train"' in events.text
    assert '"step": 2' in events.text


def test_virtual_collection_crud_scoped_tasks_and_conflict_guards(web):
    client, root, store, run = web
    assert client.get("/api/virtual-collections").json() == []
    group = client.post("/api/virtual-collections", json={"name": "Invoices"})
    assert group.status_code == 201
    group = group.json()
    key = digest(root / "red.png")
    changed = client.post(
        f"/api/virtual-collections/{group['id']}/members", json={"revision": 1, "image_ids": [key]}
    )
    assert changed.status_code == 200
    revision = changed.json()["revision"]
    stale = client.patch(
        f"/api/virtual-collections/{group['id']}", json={"name": "Old revision", "revision": 1}
    )
    assert stale.status_code == 409
    page = client.get(f"/api/virtual-collections/{group['id']}/images?revision={revision}").json()
    assert page["total"] == 1 and page["results"][0]["image_id"] == key
    queued = client.post(
        "/api/tasks/search",
        json={
            "query": "red",
            "virtual_collection": group["id"],
            "virtual_revision": revision,
            "save_run": True,
        },
    )
    result = completed(client, queued.json()["id"])
    assert result["status"] == "completed"
    assert result["virtual_collection"] == group["id"]
    assert [r["image_id"] for r in result["result"]["results"]] == [key]
    saved = client.get(f"/api/feedback/runs/{result['result']['run_id']}").json()
    assert saved["scope"]["revision"] == revision
    removed = client.request(
        "DELETE",
        f"/api/virtual-collections/{group['id']}/members",
        json={"revision": revision, "image_ids": [key]},
    )
    assert removed.status_code == 200
    renamed = client.patch(
        f"/api/virtual-collections/{group['id']}",
        json={"name": "Receipts", "revision": removed.json()["revision"]},
    )
    assert renamed.status_code == 200
    assert client.get("/api/virtual-collections").json()[0]["name"] == "Receipts"
    assert (
        client.request(
            "DELETE",
            f"/api/virtual-collections/{group['id']}",
            json={"revision": renamed.json()["revision"]},
        ).status_code
        == 200
    )
    assert (root / "red.png").is_file()
    assert client.get("/api/virtual-collections").json() == []


def test_virtual_collections_cannot_add_unknown_sources_or_cross_parent(web):
    client, root, store, run = web
    group = client.post("/api/virtual-collections", json={"name": "A"}).json()
    response = client.post(
        f"/api/virtual-collections/{group['id']}/members",
        json={"image_ids": [digest(root / "red.png"), "0" * 64], "revision": 1},
    )
    assert response.status_code == 400
    assert client.get(f"/api/virtual-collections/{group['id']}/images").json()["member_count"] == 0
    run(collection="other")
    assert (
        client.get(f"/api/virtual-collections/{group['id']}/images?collection=other").status_code
        == 400
    )
    assert (
        client.post(
            "/api/tasks/search", json={"query": "red", "virtual_collection": "../bad"}
        ).status_code
        == 422
    )


def test_duplicate_scan_task_and_review_are_non_destructive(web):
    from PIL import Image
    from realopen_lens.models.manager import digest

    client, root, store, run = web
    with Image.open(root / "red.png") as source:
        source.save(root / "alias.bmp")
    run()
    before = {p.name: digest(p) for p in root.iterdir()}
    assert client.get("/api/duplicates").json()["scanned"] is False
    response = client.post("/api/tasks/duplicates", json={"collection": "personal"})
    assert response.status_code == 202
    state = completed(client, response.json()["id"])
    assert state["status"] == "completed", state
    assert len(state["result"]["exact"]) == 1
    events = client.get(f"/api/tasks/{state['id']}/events")
    assert "finished" in events.text
    page = client.get("/api/duplicates").json()
    assert page["scanned"] and page["total"] == 1
    assert len(page["groups"][0]["results"]) == 2
    assert (
        client.post("/api/duplicates/missing/review", json={"state": "confirmed"}).status_code
        == 400
    )
    assert before == {p.name: digest(p) for p in root.iterdir()}


def test_feedback_review_download_includes_cohort_and_duplicate_audit_context(web):
    client, root, store, run = web
    launched = client.post("/api/tasks/search", json={"query": "red", "save_run": True}).json()
    state = completed(client, launched["id"])
    run_id = state["result"]["run_id"]
    key = state["result"]["results"][0]["image_id"]
    assert (
        client.put(f"/api/feedback/runs/{run_id}/{key}", json={"label": "relevant"}).status_code
        == 200
    )
    response = client.post("/api/training/snapshot", json={})
    assert response.status_code == 200, response.text
    template = response.json()["review_template"]
    audit = template["audit"]["personal"]
    assert len(audit["cohorts"]) == 1
    assert audit["runs"][0]["positive_only"]
    assert audit["duplicate_constraints"]["components"] == {}
    assert template["runs"][run_id]["split"] == ""
    assert not template["independent_holdout"]


def test_query_memory_is_applied_browsable_and_safely_cleared(web):
    client, root, store, run = web
    launch = client.post("/api/tasks/search", json={"query": "red", "save_run": True}).json()
    baseline = completed(client, launch["id"])["result"]
    saved, key = baseline["run_id"], baseline["results"][0]["image_id"]
    path = f"/api/feedback/runs/{saved}/{key}"
    changed = client.put(path, json={"label": "irrelevant", "expected_event_id": 0}).json()
    event = changed["memory"]["event_id"]
    assert client.delete(path + "?expected_event_id=0").status_code == 409
    launch = client.post("/api/tasks/search", json={"query": "  RED  "}).json()
    applied = completed(client, launch["id"])["result"]
    assert applied["feedback"]["applied"]
    assert key not in [row["image_id"] for row in applied["results"]]
    hidden = client.get("/api/feedback/hidden", params={"query": "red"}).json()
    assert hidden["total"] == 1 and hidden["results"][0]["feedback"]["event_id"] == event
    launch = client.post("/api/tasks/search", json={"query": "red", "use_feedback": False}).json()
    disabled = completed(client, launch["id"])["result"]
    assert not disabled["feedback"]["enabled"]
    assert [row["image_id"] for row in disabled["results"]] == [
        row["image_id"] for row in baseline["results"]
    ]
    assert client.delete(path + f"?expected_event_id={event}").json()["memory"]["label"] is None
    assert client.get("/api/feedback/hidden", params={"query": "red"}).json()["total"] == 0


def test_workspace_identity_and_queued_preview_import_preserve_active_index(web, tmp_path):
    client, root, store, run = web
    info = client.get("/api/workspace").json()
    assert info["directory"] == str(store.resolve())
    assert info["build"]["version"]
    assert len(info["build"]["code_sha256"]) == 64
    assert info["checkout"]["code_matches"]
    destination = tmp_path / "everyday"
    body = {"sources": [str(store)], "destination": str(destination)}
    response = client.post("/api/tasks/workspace-preview", json=body)
    assert response.status_code == 202
    task = completed(client, response.json()["id"])
    assert task["status"] == "completed", task
    plan = task["result"]
    assert plan["images"] == 2 and not destination.exists()
    response = client.post(
        "/api/tasks/workspace-import", json={**body, "plan_sha256": plan["plan_sha256"]}
    )
    task = completed(client, response.json()["id"])
    assert task["status"] == "completed", task
    assert (destination / "workspace.json").is_file()
    assert client.get("/api/workspace").json()["directory"] == info["directory"]
    assert len(client.get("/api/collections/personal/images").json()["results"]) == 2


def test_workspace_import_rejects_unreviewed_changes_and_existing_destination(web, tmp_path):
    client, root, store, run = web
    body = {"sources": [str(store)], "destination": str(tmp_path / "everyday")}
    response = client.post("/api/tasks/workspace-import", json={**body, "plan_sha256": "0" * 64})
    task = completed(client, response.json()["id"])
    assert task["status"] == "failed" and "preview changed" in task["error"]
    assert not (tmp_path / "everyday").exists()
    response = client.post(
        "/api/tasks/workspace-preview", json={"sources": [str(store)], "destination": str(store)}
    )
    task = completed(client, response.json()["id"])
    assert task["status"] == "failed" and "new destination" in task["error"]
    assert (
        client.post(
            "/api/tasks/workspace-preview", json={"sources": [], "destination": "x"}
        ).status_code
        == 422
    )


def test_empty_active_workspace_can_initialize_without_restarting(web, tmp_path):
    client, root, source, run = web
    destination = tmp_path / "empty-active"
    empty = create_app(index_dir=destination, port=8770)
    with TestClient(empty, base_url="http://127.0.0.1:8770") as active:
        active.get("/api/session")
        assert active.get("/api/collections").json() == []
        body = {"sources": [str(source)], "destination": str(destination)}
        response = active.post("/api/tasks/workspace-preview", json=body)
        task = completed(active, response.json()["id"])
        assert task["status"] == "completed", task
        response = active.post(
            "/api/tasks/workspace-import",
            json={**body, "plan_sha256": task["result"]["plan_sha256"]},
        )
        task = completed(active, response.json()["id"])
        assert task["status"] == "completed", task
        assert len(active.get("/api/collections/personal/images").json()["results"]) == 2


def test_viewer_details_full_ocr_membership_and_stable_source_identity(web, monkeypatch):
    from realopen_lens.api import viewer
    from realopen_lens.organization import collections
    from realopen_lens.storage.catalog import Catalog

    client, root, store, run = web
    listed = client.get("/api/collections/personal/images").json()
    image_id = listed["results"][0]["image_id"]
    generation = listed["generation"]
    catalog = Catalog(store, readonly=True)
    try:
        collection_id = catalog.snapshot("personal")[0]["id"]
    finally:
        catalog.close()
    folder = collections.create(store, "personal", "Folder picks")
    library = collections.create(store, "personal", "Library picks", all_folders=True)
    collections.change_members(store, "personal", folder["id"], [image_id], 1)
    response = client.get(
        "/api/image-details",
        params={
            "collection_id": collection_id,
            "collection": "old-name",
            "image_id": image_id,
            "generation": generation,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] == image_id and data["generation"] == generation
    assert data["available_paths"] and data["metadata"]["width"] > 0
    groups = {g["id"]: g for g in data["groups"]}
    assert groups[folder["id"]]["member_image_ids"] == [image_id]
    assert groups[library["id"]]["member_image_ids"] == []
    # The detail response reads the frozen OCR record, not the 300-character result preview.
    from realopen_lens.api import collections as reading

    snapshot = reading.snapshot

    def long_text(*args):
        found, manifest = snapshot(*args)
        manifest = {**manifest, "ocr": [{"image_id": image_id, "text": "مرحبا " * 100}]}
        return found, manifest

    monkeypatch.setattr(viewer, "snapshot", long_text)
    assert len(client.get("/api/image-details", params={"image_id": image_id}).json()["text"]) > 300


def test_viewer_native_actions_reject_changed_unindexed_and_untrusted_requests(web, monkeypatch):
    from realopen_lens.api import viewer

    client, root, store, run = web
    record = client.get("/api/collections/personal/images").json()["results"][0]
    called = []
    monkeypatch.setattr(
        viewer, "desktop_action", lambda path, action: called.append((path, action))
    )
    body = {"image_id": record["image_id"], "action": "reveal"}
    assert client.post("/api/image-actions", json=body).status_code == 200
    assert called == [(Path(record["paths"][0]), "reveal")]
    assert (
        client.post("/api/image-actions", json={**body, "path": "/etc/passwd"}).status_code == 422
    )
    assert client.post("/api/image-actions", json={**body, "image_id": "0" * 64}).status_code == 400
    assert (
        client.post(
            "/api/image-actions", json=body, headers={"origin": "https://example.com"}
        ).status_code
        == 403
    )
    Path(record["paths"][0]).write_bytes(b"changed")
    assert client.post("/api/image-actions", json=body).status_code == 400
    assert len(called) == 1
    client.cookies.clear()
    assert client.post("/api/image-actions", json=body).status_code == 401


@pytest.mark.parametrize(
    "platform,action,command",
    [
        ("darwin", "open", ["/usr/bin/open", "/photos/a b.png"]),
        ("darwin", "reveal", ["/usr/bin/open", "-R", "/photos/a b.png"]),
        ("win32", "reveal", ["explorer.exe", "/select,", "/photos/a b.png"]),
        ("linux", "open", ["xdg-open", "/photos/a b.png"]),
        ("linux", "reveal", ["xdg-open", "/photos"]),
    ],
)
def test_viewer_desktop_actions_use_bounded_argument_arrays(monkeypatch, platform, action, command):
    from realopen_lens.api import viewer

    calls = []
    monkeypatch.setattr(viewer.sys, "platform", platform)
    monkeypatch.setattr(viewer.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    viewer.desktop_action(Path("/photos/a b.png"), action)
    command = [*command[:-1], str(Path(command[-1]))]
    assert calls == [(command, {"check": True, "timeout": 10, "capture_output": True})]


def test_viewer_hash_verification_catches_preserved_signature_edits(web, monkeypatch):
    import os
    from realopen_lens.api import viewer

    client, root, store, run = web
    record = client.get("/api/collections/personal/images").json()["results"][0]
    path = Path(record["paths"][0])
    stat = path.stat()
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    monkeypatch.setattr(
        viewer, "desktop_action", lambda *args: pytest.fail("Changed original opened")
    )
    assert (
        client.post(
            "/api/image-actions", json={"image_id": record["image_id"], "action": "open"}
        ).status_code
        == 400
    )


def test_viewer_desktop_action_errors_are_actionable(monkeypatch):
    from realopen_lens.api import viewer

    monkeypatch.setattr(viewer.sys, "platform", "linux")

    def fail(*args, **kwargs):
        raise viewer.subprocess.TimeoutExpired("xdg-open", 10)

    monkeypatch.setattr(viewer.subprocess, "run", fail)
    with pytest.raises(ValueError, match="desktop session"):
        viewer.desktop_action(Path("/photos/image.png"), "open")


def test_viewer_membership_tracks_exact_pixel_aliases(web):
    from PIL import Image
    from realopen_lens.organization import collections, duplicates

    client, root, store, run = web
    with Image.open(root / "red.png") as source:
        source.save(root / "copy.bmp")
    run()
    duplicates.scan(store, "personal")
    listed = client.get("/api/collections/personal/images").json()["results"]
    by_name = {Path(row["paths"][0]).name: row["image_id"] for row in listed}
    first, alias = by_name["red.png"], by_name["copy.bmp"]
    group = collections.create(store, "personal", "Saved exact copy", all_folders=True)
    collections.change_members(store, "personal", group["id"], [alias], 1, all_folders=True)
    data = client.get("/api/image-details", params={"image_id": first}).json()
    assert data["groups"][0]["member_image_ids"] == [alias]
    assert (
        client.request(
            "DELETE",
            f"/api/virtual-collections/{group['id']}/members",
            json={
                "collection": "personal",
                "all_folders": True,
                "revision": data["groups"][0]["revision"],
                "image_ids": [alias],
            },
        ).status_code
        == 200
    )
    assert (
        client.get("/api/image-details", params={"image_id": first}).json()["groups"][0][
            "member_image_ids"
        ]
        == []
    )


@pytest.mark.parametrize("library", [False, True])
def test_metadata_search_tasks_interpret_natural_language_and_record_filter_context(web, library):
    import shutil
    from realopen_lens.learning.feedback import saved_search

    client, root, store, run = web
    fixture = Path(__file__).parent / "fixtures/metadata/camera-gps.heic"
    shutil.copyfile(fixture, root / "phone.heic")
    run()
    query = "photos taken with iPhone 11 Pro Max in June 2024 in Paris, France"
    task = client.post(
        "/api/tasks/search", json={"query": query, "all_folders": library, "save_run": True}
    ).json()
    result = completed(client, task["id"])
    assert result["status"] == "completed", result
    payload = result["result"]
    assert len(payload["results"]) == 1 and payload["ranking"]["kind"] == "metadata"
    assert payload["interpretation"]["visual_query"] == ""
    assert {rule["kind"] for rule in payload["interpretation"]["constraints"]} == {
        "camera",
        "date",
        "location",
    }
    saved = saved_search(payload["run_id"], index_dir=store)
    assert saved["ranking"]["query_interpretation"] == payload["interpretation"]
    details = client.get(
        "/api/image-details",
        params={"collection": "personal", "image_id": payload["results"][0]["image_id"]},
    ).json()
    assert details["metadata"]["exif"]["camera_model"] == "iPhone 11 Pro Max"
    assert details["metadata"]["exif"]["gps"]["latitude"] == pytest.approx(48.8566667)
