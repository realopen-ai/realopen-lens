"""Library collection membership and pre-retrieval scope stay independent of folders."""

import pytest

from test_library_search import (
    app as _app,
    encoders as _encoders,
    populate,
    web as _web,
    collection_fixture as _collection_fixture,
)
from test_collections import image
from test_local_api import completed
from realopen_lens.learning import feedback
from realopen_lens.models.manager import digest
from realopen_lens.organization import collections as virtual, folders, duplicates
from realopen_lens.retrieval.library import LibrarySession, LIBRARY_ID
from realopen_lens.storage.organization import CollectionConflict

app, encoders, web, collection_fixture = _app, _encoders, _web, _collection_fixture


def create(store, ids, name="Mixed collection"):
    group = virtual.create(store, "unused", name, all_folders=True)
    return virtual.change_members(
        store, "unused", group["id"], ids, group["revision"], all_folders=True
    )


def test_cross_folder_membership_browse_rename_refresh_and_unregistration(app, tmp_path):
    root, store, second, run = populate(app, tmp_path)
    keys = [digest(root / "green.png"), digest(second / "blue.png")]
    before = {p: digest(p) for path in (root, second) for p in path.iterdir()}
    group = create(store, keys)
    data = virtual.browse(store, "unused", group["id"], all_folders=True, limit=1)
    assert data["total"] == 2 and data["member_count"] == 2
    assert len(data["results"]) == 1 and data["results"][0]["sources"]
    page = virtual.browse(
        store,
        "unused",
        group["id"],
        all_folders=True,
        offset=1,
        generation=data["generation"],
        revision=group["revision"],
    )
    assert {data["results"][0]["image_id"], page["results"][0]["image_id"]} == set(keys)
    parent = virtual.parent(store, "second")
    folders.change(store, "second", parent["id"], "renamed")
    image(second / "new.png", "yellow")
    run(root=str(second), collection="renamed", siglip_precision="fp32")
    current = virtual.browse(store, "unused", group["id"], all_folders=True)
    assert {r["image_id"] for r in current["results"]} == set(keys)
    assert any(s["collection"] == "renamed" for r in current["results"] for s in r["sources"])
    with pytest.raises(CollectionConflict, match="Library index changed"):
        virtual.browse(
            store, "unused", group["id"], all_folders=True, offset=1, generation=data["generation"]
        )
    folders.change(store, "renamed", parent["id"], None)
    current = virtual.browse(store, "unused", group["id"], all_folders=True)
    assert current["total"] == 1 and current["unavailable_count"] == 1
    listed = virtual.list_collections(store, "unused", all_folders=True)[0]
    assert listed["member_count"] == 2 and listed["unavailable_count"] == 1
    assert before == {p: digest(p) for p in before}
    virtual.change_members(
        store, "unused", group["id"], [keys[1]], group["revision"], all_folders=True, remove=True
    )
    assert virtual.browse(store, "unused", group["id"], all_folders=True)["unavailable_count"] == 0


@pytest.mark.parametrize("mode", ["hybrid", "visual", "lexical", "semantic"])
def test_scope_filters_before_ranking_and_feedback_is_isolated(app, tmp_path, encoders, mode):
    root, store, second, run = populate(app, tmp_path)
    key = digest(root / "green.png")
    group = create(store, [key])
    session = LibrarySession(index_dir=store)
    try:
        result = session.search("green", mode=mode, virtual_collection=group["id"], save_run=True)
        assert [r["image_id"] for r in result["results"]] == [key]
        assert result["scope"]["collection_id"] == LIBRARY_ID
        saved = feedback.saved_search(result["run_id"], index_dir=store)
        assert saved["scope"]["members_sha256"] == result["scope"]["members_sha256"]
        feedback.change_judgment(result["run_id"], key, "irrelevant", index_dir=store)
        assert session.search("green", mode=mode, virtual_collection=group["id"])["results"] == []
        assert key in {r["image_id"] for r in session.search("green", mode=mode)["results"]}
        assert (
            session.search(
                "green", mode="lexical", virtual_collection=group["id"], hidden_only=True
            )["total"]
            == 1
        )
        renamed = virtual.rename(
            store, "unused", group["id"], "New name", group["revision"], all_folders=True
        )
        with pytest.raises(CollectionConflict):
            session.search(
                "green", virtual_collection=group["id"], virtual_revision=group["revision"]
            )
        assert renamed["revision"] > group["revision"]
    finally:
        session.close()


def test_scope_similarity_and_empty_collection_need_no_models(app, tmp_path):
    root, store, second, run = populate(app, tmp_path)
    query = digest(root / "one.png")
    key = digest(second / "blue.png")
    group = create(store, [key])
    session = LibrarySession(index_dir=store)
    try:
        result = session.similar(image_id=query, virtual_collection=group["id"])
        assert [r["image_id"] for r in result["results"]] == [key]
        empty = virtual.create(store, "unused", "Empty", all_folders=True)
        assert session.search("red", virtual_collection=empty["id"])["results"] == []
        assert session.browse(virtual_collection=empty["id"])["total"] == 0

        def mutate(event):
            if event["phase"] == "library_fusion":
                virtual.rename(
                    store, "unused", group["id"], "Changed", group["revision"], all_folders=True
                )

        with pytest.raises(CollectionConflict):
            session.browse(virtual_collection=group["id"], on_progress=mutate)
    finally:
        session.close()


def test_pixel_alias_membership_survives_original_removal_and_can_be_removed(app, tmp_path):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    other = tmp_path / "other"
    other.mkdir()
    image(other / "copy.bmp")
    run(root=str(other), collection="other")
    duplicates.scan(store, "personal")
    duplicates.scan(store, "other")
    original, alias = digest(root / "one.png"), digest(other / "copy.bmp")
    group = create(store, [original])
    same = virtual.change_members(
        store, "unused", group["id"], [alias], group["revision"], all_folders=True
    )
    assert same["revision"] == group["revision"]
    assert virtual.list_collections(store, "unused", all_folders=True)[0]["member_count"] == 1
    parent = virtual.parent(store, "personal")
    folders.change(store, "personal", parent["id"], None)
    result = virtual.browse(store, "unused", group["id"], all_folders=True)
    assert result["total"] == 1 and result["unavailable_count"] == 0
    group = virtual.change_members(
        store, "unused", group["id"], [alias], group["revision"], remove=True, all_folders=True
    )
    assert virtual.browse(store, "unused", group["id"], all_folders=True)["total"] == 0


def test_api_scope_crud_search_similarity_upload_and_namespace_isolation(web):
    client, root, store, run = web
    key = digest(root / "green.png")
    group = client.post(
        "/api/virtual-collections", json={"name": "Across folders", "all_folders": True}
    ).json()
    response = client.post(
        f"/api/virtual-collections/{group['id']}/members",
        json={
            "all_folders": True,
            "revision": group["revision"],
            "image_ids": [key],
        },
    )
    assert response.status_code == 200, response.text
    group = response.json()
    assert client.get("/api/virtual-collections?all_folders=true").json()[0]["id"] == group["id"]
    assert client.get("/api/virtual-collections?collection=personal").json() == []
    data = client.get(f"/api/virtual-collections/{group['id']}/images?all_folders=true").json()
    assert data["total"] == 1
    scope = {
        "all_folders": True,
        "virtual_collection": group["id"],
        "virtual_revision": group["revision"],
    }
    for kind, extra in [
        ("search", {"query": "green"}),
        ("similar", {"image_id": digest(root / "red.png")}),
    ]:
        response = client.post(f"/api/tasks/{kind}", json={**scope, **extra})
        assert response.status_code == 202, response.text
        task = completed(client, response.json()["id"])
        assert task["status"] == "completed", task
        assert [r["image_id"] for r in task["result"]["results"]] == [key]
    response = client.post(
        f"/api/tasks/upload?all_folders=true&virtual_collection={group['id']}"
        f"&virtual_revision={group['revision']}",
        content=(root / "red.png").read_bytes(),
    )
    task = completed(client, response.json()["id"])
    assert task["status"] == "completed", task
    assert [r["image_id"] for r in task["result"]["results"]] == [key]
    response = client.patch(
        f"/api/virtual-collections/{group['id']}",
        json={
            "all_folders": True,
            "revision": group["revision"],
            "name": "Renamed",
        },
    )
    assert response.status_code == 200
    assert client.post("/api/tasks/search", json={**scope, "query": "green"}).status_code == 409


def test_empty_workspace_library_collection_can_be_created_and_browsed(tmp_path):
    group = virtual.create(tmp_path, "unused", "Empty", all_folders=True)
    assert virtual.list_collections(tmp_path, "unused", all_folders=True)[0]["member_count"] == 0
    assert virtual.browse(tmp_path, "unused", group["id"], all_folders=True)["results"] == []


def test_library_group_cli_crud_and_scoped_searches(app, tmp_path, encoders):
    import json
    from typer.testing import CliRunner
    from realopen_lens.cli import app as cli

    root, store, other, run = populate(app, tmp_path)
    runner = CliRunner()
    suffix = ["--all-folders", "--index-dir", str(store)]
    result = runner.invoke(cli, ["groups", "create", "Mixed CLI", *suffix])
    assert result.exit_code == 0, result.output
    group = json.loads(result.output)
    key = digest(other / "blue.png")
    result = runner.invoke(cli, ["groups", "add", group["id"], key, "--revision", "1", *suffix])
    assert result.exit_code == 0, result.output
    for args in (
        ["groups", "images", group["id"]],
        ["search", "blue", "--mode", "visual", "--virtual-collection", group["id"]],
        ["similar", "--image-id", digest(root / "one.png"), "--virtual-collection", group["id"]],
    ):
        result = runner.invoke(cli, [*args, *suffix])
        assert result.exit_code == 0, result.output
        assert [r["image_id"] for r in json.loads(result.output)["results"]] == [key]
    result = runner.invoke(
        cli, ["groups", "rename", group["id"], "Changed CLI", "--revision", "2", *suffix]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli, ["groups", "list", *suffix])
    assert json.loads(result.output)[0]["name"] == "Changed CLI"
    result = runner.invoke(cli, ["groups", "delete", group["id"], "--revision", "3", *suffix])
    assert result.exit_code == 0, result.output


def test_scoped_candidates_can_retrieve_member_outside_library_top50(app, encoders):
    root, store, calls, run = app
    for i in range(65):
        image(root / f"image-{i}.png", (i * 3, i * 2, i))
    run()
    session = LibrarySession(index_dir=store)
    try:
        full = session.search("red", mode="visual", limit=100, use_feedback=False)
        key = full["results"][-1]["image_id"]
        assert key not in {r["image_id"] for r in full["results"][:50]}
        group = create(store, [key])
        actual = session.search("red", virtual_collection=group["id"], use_feedback=False)
        assert [r["image_id"] for r in actual["results"]] == [key]
    finally:
        session.close()


def test_alias_expansion_removes_more_than_500_old_identities_atomically(tmp_path, monkeypatch):
    from realopen_lens.storage.organization import OrganizationStore
    from realopen_lens.organization import library_collections

    values = [f"{i:064x}" for i in range(501)]
    group = virtual.create(tmp_path, "unused", "Aliases", all_folders=True)
    with OrganizationStore(tmp_path) as store:
        for chunk in (values[:500], values[500:]):
            group = store.members(group["id"], LIBRARY_ID, chunk, group["revision"])
    monkeypatch.setattr(
        library_collections,
        "available",
        lambda root: (set(values), dict.fromkeys(values, "same-pixels")),
    )
    changed = virtual.change_members(
        tmp_path,
        "unused",
        group["id"],
        [values[0]],
        group["revision"],
        remove=True,
        all_folders=True,
    )
    assert changed["revision"] == group["revision"] + 1
    with OrganizationStore(tmp_path, readonly=True) as store:
        assert store.snapshot(group["id"], LIBRARY_ID)["image_ids"] == []
