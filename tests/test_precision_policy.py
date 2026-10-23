"""One device policy across desktop, CLI, setup, API and Python indexing."""

import json

import pytest
from typer.testing import CliRunner

from test_collections import app as _app, image
from test_local_api import web as _web, collection_fixture as _collection_fixture
from realopen_lens.runtime import precision
from realopen_lens.indexing.service import resolve_precision

app = _app
web = _web
collection_fixture = _collection_fixture


@pytest.mark.parametrize(
    "memory,expected",
    [
        (8 * 2**30, "int8"),
        (16_000_000_000 - 1, "int8"),
        (16_000_000_000, "fp32"),
        (32 * 2**30, "fp32"),
        (None, "int8"),
    ],
)
def test_shared_ram_policy_preserves_saved_and_explicit_choices(
    tmp_path, monkeypatch, memory, expected
):
    monkeypatch.setattr(precision, "memory_bytes", lambda: memory)
    assert precision.default_precision(workspace=tmp_path) == expected
    (tmp_path / "desktop.json").write_text(
        json.dumps({"siglip_precision": "int8"}), encoding="utf-8"
    )
    assert precision.default_precision(workspace=tmp_path) == "int8"
    assert precision.default_precision("fp32", workspace=tmp_path) == "fp32"


@pytest.mark.parametrize("value", [None, "auto", [], {}])
def test_invalid_saved_profile_is_not_silently_overridden(tmp_path, value):
    (tmp_path / "desktop.json").write_text(
        json.dumps({"siglip_precision": value}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="precision"):
        precision.default_precision(workspace=tmp_path)
    assert precision.default_precision("int8", workspace=tmp_path) == "int8"


def test_existing_index_profile_wins_over_device_and_desktop_default(app, monkeypatch):
    root, store, calls, run = app
    monkeypatch.setattr(precision, "memory_bytes", lambda: 32 * 2**30)
    assert resolve_precision(store, "personal") == "fp32"
    image(root / "one.png")
    run()
    assert resolve_precision(store, "personal") == "int8"
    (store / "desktop.json").write_text('{"siglip_precision":"fp32"}', encoding="utf-8")
    assert resolve_precision(store, "personal") == "int8"
    assert resolve_precision(store, "other") == "fp32"
    assert resolve_precision(store, "personal", "fp32") == "fp32"


@pytest.mark.parametrize("memory,expected", [(8 * 2**30, "int8"), (32 * 2**30, "fp32")])
def test_standalone_api_setup_and_indexing_share_device_default(
    web, monkeypatch, tmp_path, memory, expected
):
    client, root, store, run = web
    monkeypatch.setattr(precision, "memory_bytes", lambda: memory)
    captured = []

    def submit(kind, request):
        captured.append((kind, request))
        return {"id": "fixture"}

    monkeypatch.setattr(client.app.state.manager, "submit", submit)
    assert client.get("/api/session").json()["default_siglip_precision"] == expected
    assert client.post("/api/tasks/setup", json={}).status_code == 202
    assert captured[-1][1]["siglip_precision"] == expected
    assert (
        client.post("/api/tasks/index", json={"root": str(root), "collection": "new"}).status_code
        == 202
    )
    assert captured[-1][1]["siglip_precision"] == expected
    assert (
        client.post(
            "/api/tasks/index", json={"root": str(root), "collection": "personal"}
        ).status_code
        == 202
    )
    assert captured[-1][1]["siglip_precision"] == "int8"
    (store / "desktop.json").write_text('{"siglip_precision":"int8"}', encoding="utf-8")
    assert client.get("/api/session").json()["default_siglip_precision"] == "int8"
    assert client.post("/api/tasks/setup", json={"siglip_precision": "fp32"}).status_code == 202
    assert captured[-1][1]["siglip_precision"] == "fp32"


@pytest.mark.parametrize("command", ["fetch", "verify"])
def test_model_cli_auto_profile_keeps_e5_int8(tmp_path, monkeypatch, command):
    from realopen_lens.cli import app as cli
    from realopen_lens.models import manager

    monkeypatch.setattr(precision, "default_directory", lambda: tmp_path)
    monkeypatch.setattr(precision, "memory_bytes", lambda: 32 * 2**30)
    selected = []

    def capture(root, base, *args, **kwargs):
        selected.append((base, kwargs.get("siglip_precision", args[0] if args else None)))

    monkeypatch.setattr(manager, command, capture)
    runner = CliRunner()
    assert runner.invoke(cli, ["models", command]).exit_code == 0
    assert selected[-1] == ("int8", "fp32")
    assert runner.invoke(cli, ["models", command, "--siglip-precision", "int8"]).exit_code == 0
    assert selected[-1] == ("int8", "int8")
    assert runner.invoke(cli, ["models", command, "--precision", "int8"]).exit_code == 0
    assert selected[-1] == ("int8", "int8")


def test_python_index_boundary_resolves_profile_before_worker_launch(tmp_path, monkeypatch):
    from realopen_lens.indexing import service

    monkeypatch.setattr(precision, "memory_bytes", lambda: 32 * 2**30)

    class StopLaunch(Exception):
        pass

    request = []

    def capture(path, data):
        request.append(json.loads(data))
        raise StopLaunch()

    monkeypatch.setattr(service, "atomic_bytes", capture)
    with pytest.raises(StopLaunch):
        service.index_collection(tmp_path, index_dir=tmp_path / "index")
    assert request[-1]["siglip_precision"] == "fp32"
    with pytest.raises(StopLaunch):
        service.index_collection(tmp_path, index_dir=tmp_path / "index", siglip_precision="int8")
    assert request[-1]["siglip_precision"] == "int8"


def test_interrupted_initial_index_preserves_its_explicit_profile(tmp_path, monkeypatch):
    from realopen_lens.storage.catalog import Catalog

    monkeypatch.setattr(precision, "memory_bytes", lambda: 32 * 2**30)
    catalog = Catalog(tmp_path)
    try:
        folder = catalog.ensure_collection("pending", tmp_path / "photos")
        job = catalog.create_job(folder["id"], {"spec": {"siglip_precision": "int8"}})
        catalog.finish(job, "interrupted")
    finally:
        catalog.close()
    assert resolve_precision(tmp_path, "pending") == "int8"
