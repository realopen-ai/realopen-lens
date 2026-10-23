"""Automatic refresh through real discovery, publication, task state and local API."""

import threading
import json
from time import monotonic

import pytest

from test_local_api import web as _web, collection_fixture as _collection_fixture, completed
from test_collections import image
from realopen_lens.api import tasks
from realopen_lens.api.refresh import FolderRefresh
from realopen_lens.indexing import changes, service
from realopen_lens.storage.catalog import Catalog

web = _web
collection_fixture = _collection_fixture


@pytest.fixture(autouse=True)
def deterministic_monitor(monkeypatch):
    monkeypatch.setattr(FolderRefresh, "start", lambda self: None)


def configure(client, enabled=True, interval=60):
    folder = client.get("/api/collections").json()[0]
    response = client.post(
        "/api/collections/auto-refresh",
        json={
            "collection": folder["name"],
            "collection_id": folder["id"],
            "enabled": enabled,
            "interval_seconds": interval,
        },
    )
    assert response.status_code == 200, response.text
    return folder, response.json()


def drive(client, instant):
    client.app.state.manager.auto_refresh.check_once(now=instant)


def indexer(monkeypatch, run):
    calls = []

    def index(path, **options):
        calls.append(options)
        options["on_progress"]({"phase": "scanning", "files": 3})
        return run(
            **{
                k: v
                for k, v in options.items()
                if k
                in {
                    "collection",
                    "languages",
                    "siglip_precision",
                    "expected_source_identity",
                    "expected_collection_id",
                }
            }
        )

    monkeypatch.setattr(tasks, "index_collection", index)
    return calls


def test_auto_refresh_is_opt_in_and_does_not_reindex_unchanged_files(web, monkeypatch):
    client, root, store, run = web
    calls = indexer(monkeypatch, run)
    drive(client, monotonic() + 100)
    assert not client.get("/api/tasks").json()
    folder, settings = configure(client)
    assert settings["enabled"] and settings["interval_seconds"] == 60
    drive(client, monotonic() + 100)
    assert not calls
    assert client.get("/api/collections").json()[0]["auto_refresh"]["state"] == "waiting"
    # Disabling must be possible even after a drive disappears.
    root.rename(root.with_name("offline"))
    assert configure(client, enabled=False)[1]["enabled"] is False


@pytest.mark.parametrize("precision", ["int8", "fp32"])
def test_bursts_batch_and_auto_refresh_reuses_index_settings_and_stages(
    web, monkeypatch, precision
):
    client, root, store, run = web
    name = f"auto-{precision}"
    run(collection=name, siglip_precision=precision, languages="eng+fra")
    calls = indexer(monkeypatch, run)
    folder, _ = configure(client)
    instant = monotonic() + 100
    image(root / "one.png", "blue")
    drive(client, instant)
    assert client.get("/api/collections").json()[0]["auto_refresh"]["state"] == "settling"
    image(root / "two.png", "white")
    (root / "green.png").unlink()
    drive(client, instant + 10)
    assert not calls
    drive(client, instant + 20)
    task = client.get("/api/tasks").json()[0]
    assert task["origin"] == "automatic" and task["collection_id"] == folder["id"]
    result = completed(client, task["id"])
    assert result["status"] == "completed", result
    assert result["result"]["reused_images"] == 1
    assert result["result"]["removed_files"] == 1
    assert calls[0]["languages"] == "eng+fra" and calls[0]["siglip_precision"] == precision
    assert client.get(f"/api/collections/{name}/images").json()["total"] == 3
    assert "event: progress" in client.get(f"/api/tasks/{task['id']}/events").text
    drive(client, instant + 200)
    assert len(calls) == 1


def test_disconnected_or_replaced_root_does_not_publish_deletions(web, monkeypatch):
    client, root, store, run = web
    calls = indexer(monkeypatch, run)
    folder, _ = configure(client)
    offline = root.with_name("offline")
    root.rename(offline)
    instant = monotonic() + 100
    drive(client, instant)
    assert not calls
    assert client.get("/api/collections").json()[0]["auto_refresh"]["state"] == "unavailable"
    root.mkdir()  # An empty mount point or replacement is not the original directory.
    drive(client, instant + 60)
    assert not calls
    assert "identity changed" in client.get("/api/collections").json()[0]["auto_refresh"]["error"]
    # Changing only the polling interval must not authorize a replacement root.
    configure(client, interval=60)
    drive(client, instant + 61)
    assert not calls
    root.rmdir()
    offline.rename(root)
    drive(client, instant + 120)
    assert client.get("/api/collections").json()[0]["generation"] == folder["generation"]
    assert client.get("/api/collections/personal/images").json()["total"] == 2


def test_read_failure_and_scan_limit_never_schedule_an_index(web, monkeypatch):
    client, root, store, run = web
    calls = indexer(monkeypatch, run)
    configure(client)
    monkeypatch.setattr(changes, "MAX_ENTRIES", 1)
    instant = monotonic() + 100
    drive(client, instant)
    assert "limit" in client.get("/api/collections").json()[0]["auto_refresh"]["error"]
    monkeypatch.setattr(changes, "MAX_ENTRIES", 100_000)

    def denied(*args, **kwargs):
        raise PermissionError("Unreadable subtree")

    monkeypatch.setattr(changes.os, "scandir", denied)
    drive(client, instant + 60)
    assert not calls
    assert "Unreadable subtree" in client.get("/api/collections").json()[0]["auto_refresh"]["error"]


def test_metadata_scan_matches_index_and_ignores_store_and_git(web):
    client, root, store, run = web
    folder, _ = configure(client)
    ignored = root / ".git"
    ignored.mkdir()
    image(ignored / "git.png")
    # Discovery must stay metadata-only; hashing image bytes is the indexer's job.
    assert changes.observed(
        root, store, changes.source_identity(root), lambda: False
    ) == changes.indexed(store, folder["id"])


def test_cancel_stays_paused_instead_of_automatically_restarting(web, monkeypatch):
    client, root, store, run = web
    configure(client)
    started, release = threading.Event(), threading.Event()

    def index(path, **options):
        started.set()
        assert release.wait(5)
        options["on_progress"]({"phase": "vision"})
        pytest.fail("Cancelled refresh must stop")

    monkeypatch.setattr(tasks, "index_collection", index)
    image(root / "new.png")
    instant = monotonic() + 100
    drive(client, instant)
    drive(client, instant + 10)
    assert started.wait(5)
    task = client.get("/api/tasks").json()[0]
    try:
        client.post(f"/api/tasks/{task['id']}/cancel")
    finally:
        release.set()
    assert completed(client, task["id"])["status"] == "cancelled"
    drive(client, instant + 200)
    settings = client.get("/api/collections").json()[0]["auto_refresh"]
    assert not settings["enabled"] and "cancellation" in settings["error"]
    assert len(client.get("/api/tasks").json()) == 1


def test_failures_pause_while_foreground_jobs_defer_monitoring(web, monkeypatch):
    client, root, store, run = web
    configure(client)
    image(root / "new.png")
    manager = client.app.state.manager
    instant = monotonic() + 100
    with manager.lock:
        manager.tasks["foreground"] = {"status": "running"}
    drive(client, instant)
    assert (
        manager.auto_refresh.states[next(iter(manager.auto_refresh.settings))]["state"] == "waiting"
    )
    with manager.lock:
        manager.tasks.pop("foreground")

    def broken(*args, **kwargs):
        raise ValueError("Models unavailable")

    monkeypatch.setattr(tasks, "index_collection", broken)
    drive(client, instant)
    drive(client, instant + 10)
    task = client.get("/api/tasks").json()[0]
    assert completed(client, task["id"])["status"] == "failed"
    settings = client.get("/api/collections").json()[0]["auto_refresh"]
    assert not settings["enabled"] and "Models unavailable" in settings["error"]


def test_settings_survive_rename_restart_and_are_removed_with_folder(web, monkeypatch):
    client, root, store, run = web
    folder, _ = configure(client, interval=300)
    manager = client.app.state.manager
    response = client.post(
        "/api/collections/rename",
        json={"collection": folder["name"], "collection_id": folder["id"], "name": "My photos"},
    )
    assert response.status_code == 200
    restored = FolderRefresh(manager)
    assert restored.describe(folder["id"])["enabled"]
    assert restored.describe(folder["id"])["interval_seconds"] == 300
    response = client.post(
        "/api/collections/remove", json={"collection": "My photos", "collection_id": folder["id"]}
    )
    assert response.status_code == 200
    assert folder["id"] not in FolderRefresh(manager).settings
    run()
    assert not client.get("/api/collections").json()[0]["auto_refresh"]["enabled"]


def test_restart_detects_changes_made_while_offline(web, monkeypatch):
    client, root, store, run = web
    folder, _ = configure(client)
    manager = client.app.state.manager
    image(root / "new.png", "blue")
    calls = indexer(monkeypatch, run)
    manager.auto_refresh = FolderRefresh(manager)
    instant = monotonic() + 100
    drive(client, instant)
    drive(client, instant + 10)
    task = client.get("/api/tasks").json()[0]
    assert completed(client, task["id"])["status"] == "completed"
    assert len(calls) == 1


def test_stale_configuration_and_invalid_intervals_are_rejected(web):
    client, root, store, run = web
    folder = client.get("/api/collections").json()[0]
    for interval in (0, 29, 3601):
        assert (
            client.post(
                "/api/collections/auto-refresh",
                json={"collection_id": folder["id"], "enabled": True, "interval_seconds": interval},
            ).status_code
            == 422
        )
    assert (
        client.post(
            "/api/collections/auto-refresh", json={"collection_id": "f" * 32, "enabled": True}
        ).status_code
        == 409
    )


def test_indexer_rechecks_root_before_publishing_and_rejects_stale_ids(
    collection_fixture, monkeypatch
):
    root, store, calls, run = collection_fixture
    image(root / "original.png")
    initial = run()
    catalog = Catalog(store)
    key = catalog.collection("personal")["id"]
    catalog.close()
    with pytest.raises(ValueError, match="collection changed"):
        run(expected_collection_id="f" * 32)
    identity = changes.source_identity(root)
    original = service.discover

    def replaced(*args, **kwargs):
        original(*args, **kwargs)
        root.rename(root.with_name("old"))
        root.mkdir()

    monkeypatch.setattr(service, "discover", replaced)
    with pytest.raises(ValueError, match="identity changed"):
        run(expected_source_identity=identity, expected_collection_id=key)
    catalog = Catalog(store)
    assert catalog.collection("personal")["active_generation"] == initial.generation
    catalog.close()


def test_corrupt_settings_do_not_break_gallery_and_can_be_recovered_with_backup(web):
    client, root, store, run = web
    manager = client.app.state.manager
    path = store / "folder-refresh.json"
    path.write_bytes(b"not valid JSON")
    manager.auto_refresh = FolderRefresh(manager)
    settings = client.get("/api/collections").json()[0]["auto_refresh"]
    assert not settings["enabled"] and "could not be read" in settings["error"]
    configure(client)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    assert next(store.glob("folder-refresh.invalid-*.json")).read_bytes() == b"not valid JSON"


def test_scan_is_bounded_and_cancellable_without_opening_images(web, monkeypatch):
    client, root, store, run = web
    identity = changes.source_identity(root)
    with pytest.raises(InterruptedError, match="stopped"):
        changes.observed(root, store, identity, lambda: True)
    monkeypatch.setattr(changes, "SCAN_SECONDS", -1)
    with pytest.raises(ValueError, match="limit"):
        changes.observed(root, store, identity, lambda: False)


def test_background_thread_discovers_and_publishes_without_manual_refresh(web, monkeypatch):
    import time
    from realopen_lens.api import refresh

    client, root, store, run = web
    configure(client)
    calls = indexer(monkeypatch, run)
    monkeypatch.setattr(refresh, "SETTLE_SECONDS", 0)
    image(root / "background.png", "blue")
    monitor = client.app.state.manager.auto_refresh
    # Use the real loop with a fast stop-event wait, keeping scheduling deterministic elsewhere.
    original_wait = monitor.stop.wait
    monkeypatch.setattr(monitor.stop, "wait", lambda timeout: original_wait(0.01))
    monitor.thread = threading.Thread(target=monitor._loop, daemon=True)
    monitor.thread.start()
    try:
        deadline = time.monotonic() + 5
        while not client.get("/api/tasks").json():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        task = client.get("/api/tasks").json()[0]
        assert completed(client, task["id"])["status"] == "completed"
        assert client.get("/api/collections/personal/images").json()["total"] == 3
        assert len(calls) == 1
    finally:
        monitor.close()
    assert not monitor.thread.is_alive()


def test_settings_write_failure_does_not_enable_unpersisted_monitoring(web, monkeypatch):
    client, root, store, run = web
    folder = client.get("/api/collections").json()[0]
    monitor = client.app.state.manager.auto_refresh

    def denied():
        raise PermissionError("Settings cannot be written")

    monkeypatch.setattr(monitor, "_save", denied)
    response = client.post(
        "/api/collections/auto-refresh",
        json={
            "collection_id": folder["id"],
            "enabled": True,
        },
    )
    assert response.status_code == 400
    assert not monitor.describe(folder["id"])["enabled"]


def test_shutdown_keeps_automatic_refresh_enabled_for_next_start(web, monkeypatch):
    client, root, store, run = web
    folder, _ = configure(client)
    started, release = threading.Event(), threading.Event()

    def index(path, **options):
        started.set()
        assert release.wait(5)
        options["on_progress"]({"phase": "vision"})
        pytest.fail("Shutdown should interrupt owned automatic work")

    monkeypatch.setattr(tasks, "index_collection", index)
    image(root / "new.png", "blue")
    instant = monotonic() + 100
    drive(client, instant)
    drive(client, instant + 10)
    assert started.wait(5)
    manager = client.app.state.manager
    shutting_down = threading.Event()
    close_monitor = manager.auto_refresh.close

    def close():
        shutting_down.set()
        close_monitor()

    monkeypatch.setattr(manager.auto_refresh, "close", close)
    closer = threading.Thread(target=manager.close)
    closer.start()
    try:
        assert shutting_down.wait(5)
        # close marks work cancelled before waiting for its executor.
        with manager.lock:
            assert manager.closed
    finally:
        release.set()
        closer.join(5)
    assert not closer.is_alive()
    assert FolderRefresh(manager).describe(folder["id"])["enabled"]
    assert manager.list()[0]["status"] == "cancelled"


def test_external_cli_writer_defers_automatic_work_without_disabling_it(web, monkeypatch):
    from realopen_lens.runtime.jobs import writer_lock

    client, root, store, run = web
    configure(client)
    calls = indexer(monkeypatch, run)
    image(root / "new.png", "blue")
    instant = monotonic() + 100
    drive(client, instant)
    with writer_lock(store):
        drive(client, instant + 10)
    assert not calls and not client.get("/api/tasks").json()
    assert client.get("/api/collections").json()[0]["auto_refresh"]["enabled"]
    drive(client, instant + 100)
    task = client.get("/api/tasks").json()[0]
    assert completed(client, task["id"])["status"] == "completed"


def test_task_discovery_can_omit_results_without_mutating_saved_task(web):
    client, root, store, run = web
    response = client.post("/api/tasks/search", json={"query": "red"})
    result = completed(client, response.json()["id"])
    assert result["result"]["results"]
    summary = client.get("/api/tasks", params={"summary": True}).json()[0]
    assert summary["result"] is None and summary["revision"] == result["revision"]
    assert client.get("/api/tasks/" + result["id"]).json()["result"] == result["result"]


def test_interrupted_task_recovery_advances_revision_and_retains_automation(web):
    from time import time

    client, root, store, run = web
    folder, _ = configure(client)
    manager = client.app.state.manager
    manager._persist(
        {
            "id": "f" * 32,
            "collection": "personal",
            "collection_id": folder["id"],
            "kind": "index",
            "origin": "automatic",
            "status": "running",
            "phase": "vision",
            "revision": 3,
            "created_at": time(),
            "progress": {},
            "error": None,
            "result": None,
        }
    )
    recovered = tasks.TaskManager(store, store / "models")
    try:
        task = recovered.snapshot("f" * 32)
        assert task["status"] == "interrupted" and task["revision"] == 4
        assert recovered.auto_refresh.describe(folder["id"])["enabled"]
    finally:
        recovered.close()
