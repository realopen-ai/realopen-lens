"""Snapshot paging retains the ordinary head and rejects stale scope/source state."""

from copy import deepcopy

import pytest

from realopen_lens.api.search_pages import SearchPages, POOL_LIMIT
from realopen_lens.storage.organization import CollectionConflict
from test_local_api import web as _web, collection_fixture as _collection_fixture, completed

web = _web
collection_fixture = _collection_fixture


def row(key, path, **extra):
    return {"image_id": key, "paths": [str(path)], "duplicate_ids": [], **extra}


def test_snapshot_preserves_head_deduplicates_and_never_retrieves_on_continuation(
    tmp_path, monkeypatch
):
    path = tmp_path / "image.png"
    path.write_bytes(b"fixture")
    pages = SearchPages(tmp_path)
    monkeypatch.setattr(pages, "identity", lambda request: "fixed")
    calls = []

    def retrieve(request):
        calls.append(deepcopy(request))
        values = (
            [row("a", path), row("b", path)]
            if len(calls) == 1
            else [
                row("c", path),
                row("b", path),
                row("d", path, duplicate_ids=["a"]),
                row("e", path),
            ]
        )
        return {"results": values, "run_id": None, "ranking": {"kind": "logistic"}}

    first = pages.create({"limit": 2}, retrieve)
    assert [r["image_id"] for r in first["results"]] == ["a", "b"]
    assert first["total"] == 4
    assert calls[1]["deep_search"] and calls[1]["limit"] == POOL_LIMIT
    second = pages.page(first["next_cursor"])
    assert [r["image_id"] for r in second["results"]] == ["c", "e"]
    assert [r["rank"] for r in second["results"]] == [3, 4]
    assert second["next_cursor"] is None and len(calls) == 2
    second["results"].clear()
    assert len(pages.page(first["next_cursor"])["results"]) == 2
    path.write_bytes(b"source changed")
    with pytest.raises(CollectionConflict, match="source changed"):
        pages.page(first["next_cursor"])


def test_snapshot_expiry_eviction_cap_and_scope_change(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    path.touch()
    pages = SearchPages(tmp_path)
    identity = ["old"]
    monkeypatch.setattr(pages, "identity", lambda request: identity[0])

    def retrieve(request):
        return {"results": [row(str(i), path) for i in range(1001)]}

    first = pages.create({"limit": 50}, retrieve)
    assert first["total"] == 1000 and first["truncated"]
    with pytest.raises(ValueError, match="cursor"):
        pages.page("../../bad")
    identity[0] = "new"
    with pytest.raises(CollectionConflict, match="scope changed"):
        pages.page(first["next_cursor"])
    identity[0] = "old"
    for _ in range(8):
        pages.create({"limit": 50}, retrieve)
    assert len(pages.snapshots) == 8
    with pytest.raises(CollectionConflict, match="expired"):
        pages.page(first["next_cursor"])
    latest = next(reversed(pages.snapshots.values()))
    latest["created"] -= 1801
    pages._prune()
    assert len(pages.snapshots) == 7


@pytest.mark.parametrize("library", [False, True])
def test_production_text_pages_keep_saved_feedback_and_detect_new_generation(web, library):
    client, root, store, run = web
    if library:
        run(collection="other", siglip_precision="fp32")
    request = {
        "query": "red green",
        "mode": "lexical",
        "limit": 1,
        "save_run": True,
        "paginate": True,
        "all_folders": library,
    }
    task = client.post("/api/tasks/search", json=request)
    state = completed(client, task.json()["id"])
    assert state["status"] == "completed", state
    first = state["result"]
    from realopen_lens.storage.feedback import FeedbackStore

    with FeedbackStore(store, readonly=True) as records:
        assert len(records.runs(100)) == 1
    assert first["total"] == 2 and first["next_cursor"]
    second = client.get("/api/search-pages", params={"cursor": first["next_cursor"]}).json()
    assert second["run_id"] == first["run_id"]
    key = second["results"][0]["image_id"]
    assert key != first["results"][0]["image_id"] and second["results"][0]["rank"] == 2
    assert (
        client.put(
            f"/api/feedback/runs/{first['run_id']}/{key}", json={"label": "irrelevant"}
        ).status_code
        == 200
    )
    hidden = client.get("/api/search-pages", params={"cursor": first["next_cursor"]}).json()
    assert hidden["results"] == []
    image = root / "red.png"
    image.touch()
    assert (
        client.get("/api/search-pages", params={"cursor": first["next_cursor"]}).status_code == 409
    )


@pytest.mark.parametrize("change", ["membership", "virtual_removal", "folder_removal"])
def test_virtual_membership_revision_invalidates_search_page(web, change):
    client, root, store, run = web
    images = client.get("/api/collections/personal/images").json()["results"]
    group = client.post("/api/virtual-collections", json={"name": "Selected"}).json()
    updated = client.post(
        f"/api/virtual-collections/{group['id']}/members",
        json={"revision": group["revision"], "image_ids": [r["image_id"] for r in images]},
    ).json()
    state = completed(
        client,
        client.post(
            "/api/tasks/search",
            json={
                "query": "red green",
                "mode": "lexical",
                "limit": 1,
                "paginate": True,
                "virtual_collection": group["id"],
                "virtual_revision": updated["revision"],
            },
        ).json()["id"],
    )
    assert state["status"] == "completed", state
    cursor = state["result"]["next_cursor"]
    assert cursor
    if change == "membership":
        assert (
            client.request(
                "DELETE",
                f"/api/virtual-collections/{group['id']}/members",
                json={"revision": updated["revision"], "image_ids": [images[0]["image_id"]]},
            ).status_code
            == 200
        )
    elif change == "virtual_removal":
        assert (
            client.request(
                "DELETE",
                f"/api/virtual-collections/{group['id']}",
                json={"revision": updated["revision"]},
            ).status_code
            == 200
        )
    else:
        folder = client.get("/api/collections").json()[0]
        assert (
            client.post(
                "/api/collections/remove",
                json={"collection": folder["name"], "collection_id": folder["id"]},
            ).status_code
            == 200
        )
    assert client.get("/api/search-pages", params={"cursor": cursor}).status_code == 409


def test_image_similarity_paginates_and_excludes_query_image(web):
    from test_collections import image

    client, root, store, run = web
    image(root / "blue.png", "blue")
    run()
    query = client.get("/api/collections/personal/images").json()["results"][0]["image_id"]
    state = completed(
        client,
        client.post(
            "/api/tasks/similar",
            json={
                "image_id": query,
                "limit": 1,
                "paginate": True,
                "save_run": True,
            },
        ).json()["id"],
    )
    assert state["status"] == "completed", state
    first = state["result"]
    assert first["total"] == 2
    last = client.get("/api/search-pages", params={"cursor": first["next_cursor"]}).json()
    assert query not in {row["image_id"] for row in first["results"] + last["results"]}
    assert len({row["image_id"] for row in first["results"] + last["results"]}) == 2


def test_deep_labels_export_without_claiming_original_budget_features(web, tmp_path):
    import json
    from realopen_lens.learning.feedback import export_feedback
    from realopen_lens.storage.feedback import FeedbackStore

    client, root, store, run = web
    state = completed(
        client,
        client.post(
            "/api/tasks/search",
            json={
                "query": "red green",
                "limit": 1,
                "paginate": True,
                "save_run": True,
            },
        ).json()["id"],
    )
    assert state["status"] == "completed", state
    first = state["result"]
    second = client.get("/api/search-pages", params={"cursor": first["next_cursor"]}).json()
    head, tail = first["results"][0]["image_id"], second["results"][0]["image_id"]
    with FeedbackStore(store, readonly=True) as records:
        saved = records.run(first["run_id"])
        assert set(saved["candidates"]) == {head}
        assert saved["ranking"]["candidate_depths"] == {"visual": 50, "semantic": 50, "lexical": 50}
        assert saved["ranking"]["pagination"]["head_count"] == 1
    assert (
        client.put(
            f"/api/feedback/runs/{first['run_id']}/{tail}", json={"label": "relevant"}
        ).status_code
        == 200
    )
    output = tmp_path / "export.jsonl"
    export_feedback(output, index_dir=store)
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["image_id"] == tail and exported["candidate_schema"] is None
    assert exported["candidate"] is None and exported["judgment"] == "relevant"
