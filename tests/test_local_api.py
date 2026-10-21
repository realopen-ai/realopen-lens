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
