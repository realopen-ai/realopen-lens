"""Saved-folder refresh through the production catalog and local API."""

import pytest

from test_local_api import (
    web as _web,
    collection_fixture as _collection_fixture,
    completed,
)
from test_collections import image
from realopen_lens.api import tasks

web = _web
collection_fixture = _collection_fixture


def identity(client):
    row = client.get("/api/collections").json()[0]
    return {"collection": row["name"], "collection_id": row["id"]}


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


def test_refresh_missing_source_is_reported_without_unregistering(web):
    client, root, store, run = web
    body = identity(client)
    root.rename(root.with_name("disconnected"))
    response = client.post("/api/collections/refresh", json=body)
    assert response.status_code == 400 and "unavailable" in response.text
    assert client.get("/api/collections").json()[0]["id"] == body["collection_id"]
