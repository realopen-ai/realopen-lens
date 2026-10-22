"""Folder lifecycle regressions through the production catalog and local API."""

import json
import threading

import pytest


from test_local_api import web as _web, collection_fixture as _collection_fixture, completed
from test_collections import image
from realopen_lens.api import tasks
from realopen_lens.learning import registry
from realopen_lens.organization import duplicates
from realopen_lens.runtime.jobs import writer_lock
from realopen_lens.storage.catalog import Catalog

web = _web
collection_fixture = _collection_fixture


def identity(client):
    row = client.get("/api/collections").json()[0]
    return {"collection": row["name"], "collection_id": row["id"]}


def test_rename_preserves_generation_feedback_groups_reviews_and_activation(web, monkeypatch):
    client, root, store, run = web
    body = identity(client)
    task = client.post("/api/tasks/search", json={"query": "red", "save_run": True}).json()
    result = completed(client, task["id"])["result"]
    saved, key = result["run_id"], result["results"][0]["image_id"]
    assert (
        client.put(f"/api/feedback/runs/{saved}/{key}", json={"label": "irrelevant"}).status_code
        == 200
    )
    group = client.post("/api/virtual-collections", json={"name": "Favorites"}).json()
    assert (
        client.post(
            f"/api/virtual-collections/{group['id']}/members",
            json={"revision": 1, "image_ids": [key]},
        ).status_code
        == 200
    )
    duplicates.scan(store, "personal")
    review = store / "duplicates" / "reviews" / (body["collection_id"] + ".json")
    review.parent.mkdir(exist_ok=True)
    review.write_text('{"reviewed-pair":"confirmed"}', encoding="utf-8")
    before = {path: path.read_bytes() for path in root.iterdir()}
    version = "a" * 32
    state = store / "rerankers" / "state.json"
    state.parent.mkdir(exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "version": 1,
                "active": {"personal": version},
                "history": [{"collection": "personal", "to": version}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "compatible", lambda *args: store / "model.json")
    response = client.post("/api/collections/rename", json={**body, "name": "Mes photos صور"})
    assert response.status_code == 200, response.text
    renamed = client.get("/api/collections").json()[0]
    assert renamed["id"] == body["collection_id"]
    assert client.get("/api/collections/Mes photos صور/images").json()["total"] == 2
    assert (
        client.get("/api/virtual-collections", params={"collection": renamed["name"]}).json()[0][
            "id"
        ]
        == group["id"]
    )
    assert client.get(f"/api/feedback/runs/{saved}").json()["judgments"][0]["label"] == "irrelevant"
    hidden = client.get(
        "/api/feedback/hidden", params={"collection": renamed["name"], "query": "red"}
    ).json()
    assert hidden["total"] == 1
    assert review.read_text(encoding="utf-8") == '{"reviewed-pair":"confirmed"}'
    assert registry.active(store, renamed["name"]) == store / "model.json"
    assert client.get("/api/training/models").json()["active"] == {renamed["name"]: version}
    assert registry.state(store)["history"][0]["collection_id"] == body["collection_id"]
    assert {path: path.read_bytes() for path in root.iterdir()} == before
    # Reusing the old display name must not inherit the migrated activation.
    catalog = Catalog(store)
    catalog.ensure_collection("personal", root)
    catalog.ensure_collection(body["collection_id"], root)
    catalog.close()
    assert registry.active(store, "personal") is None
    assert registry.active(store, body["collection_id"]) is None


def test_remove_unregisters_only_target_and_stale_requests_cannot_affect_replacement(web):
    client, root, store, run = web
    body = identity(client)
    before = {path: path.read_bytes() for path in root.iterdir()}
    catalog = Catalog(store)
    other = catalog.ensure_collection("other", root)
    catalog.close()
    response = client.post("/api/collections/remove", json=body)
    assert response.status_code == 200, response.text
    assert client.get("/api/collections").json()[0]["id"] == other["id"]
    assert client.get("/api/collections/personal/images").status_code == 400
    assert {path: path.read_bytes() for path in root.iterdir()} == before
    run()
    for action in ("remove", "refresh", "rename"):
        payload = {**body, **({"name": "bad"} if action == "rename" else {})}
        assert client.post(f"/api/collections/{action}", json=payload).status_code == 409
    assert client.get("/api/collections/personal/images").json()["total"] == 2


@pytest.mark.parametrize("precision", ["int8", "fp32"])
def test_refresh_uses_saved_settings_and_publishes_additions_and_removals(
    web, monkeypatch, precision
):
    client, root, store, run = web
    name = f"refresh-{precision}"
    run(collection=name, siglip_precision=precision, languages="eng+fra")
    row = next(row for row in client.get("/api/collections").json() if row["name"] == name)
    body = {"collection": name, "collection_id": row["id"]}
    image(root / "new.png", "blue")
    (root / "green.png").unlink()
    captured = {}

    def index(path, **options):
        captured.update(options)
        options["on_progress"]({"phase": "scanning", "files": 2})
        return run(
            collection=options["collection"],
            siglip_precision=options["siglip_precision"],
            languages=options["languages"],
        )

    monkeypatch.setattr(tasks, "index_collection", index)
    response = client.post("/api/collections/refresh", json=body)
    assert response.status_code == 202, response.text
    result = completed(client, response.json()["id"])
    assert result["status"] == "completed", result
    assert captured["languages"] == "eng+fra" and captured["siglip_precision"] == precision
    rows = client.get(f"/api/collections/{name}/images").json()["results"]
    paths = [path for row in rows for path in row["paths"]]
    assert any(path.endswith("new.png") for path in paths)
    assert not any(path.endswith("green.png") for path in paths)
    assert result["result"]["reused_images"] == 1


def test_mutations_reject_busy_jobs_conflicts_and_external_writer(web, monkeypatch):
    client, root, store, run = web
    body = identity(client)
    started, release = threading.Event(), threading.Event()

    def index(path, **options):
        started.set()
        assert release.wait(5)
        return run()

    monkeypatch.setattr(tasks, "index_collection", index)
    task = client.post("/api/collections/refresh", json=body).json()
    assert started.wait(5)
    try:
        for action in ("refresh", "remove", "rename"):
            payload = {**body, **({"name": "new"} if action == "rename" else {})}
            assert client.post(f"/api/collections/{action}", json=payload).status_code == 409
    finally:
        release.set()
    assert completed(client, task["id"])["status"] == "completed"
    with writer_lock(store):
        assert client.post("/api/collections/remove", json=body).status_code == 400
    catalog = Catalog(store)
    catalog.ensure_collection("other", root)
    catalog.close()
    assert client.post("/api/collections/rename", json={**body, "name": "other"}).status_code == 409
    for name in ("  ", "bad\nname"):
        assert (
            client.post("/api/collections/rename", json={**body, "name": name}).status_code == 400
        )


def test_refresh_missing_source_is_reported_without_unregistering(web):
    client, root, store, run = web
    body = identity(client)
    root.rename(root.with_name("disconnected"))
    response = client.post("/api/collections/refresh", json=body)
    assert response.status_code == 400 and "unavailable" in response.text
    assert client.get("/api/collections").json()[0]["id"] == body["collection_id"]


def test_removal_retains_exportable_feedback_without_leaking_to_readded_folder(web, tmp_path):
    from realopen_lens.learning import feedback

    client, root, store, run = web
    body = identity(client)
    response = client.post("/api/tasks/search", json={"query": "red", "save_run": True})
    result = completed(client, response.json()["id"])["result"]
    saved, key = result["run_id"], result["results"][0]["image_id"]
    assert (
        client.put(f"/api/feedback/runs/{saved}/{key}", json={"label": "irrelevant"}).status_code
        == 200
    )
    assert client.post("/api/collections/remove", json=body).status_code == 200
    assert feedback.export_feedback(tmp_path / "history.jsonl", index_dir=store) == 1
    run()
    assert client.get("/api/feedback/hidden", params={"query": "red"}).json()["total"] == 0
    assert client.get(f"/api/feedback/runs/{saved}").json()["judgments"][0]["label"] == "irrelevant"
