"""Setup must report missing prerequisites, not confuse successful checks with readiness."""

import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from realopen_lens.runtime import readiness


@pytest.fixture
def device(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    content = b"a small pinned model fixture"
    selected = []

    def artifacts(precision, siglip):
        selected.append((precision, siglip))
        return [
            (
                "siglip2",
                "vision",
                {
                    "path": "vision.onnx",
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
            )
        ]

    monkeypatch.setattr(readiness, "artifacts", artifacts)
    monkeypatch.setattr(readiness, "artifact_path", lambda root, *args: root / "vision.onnx")
    monkeypatch.setattr(
        readiness,
        "installed_ocr",
        lambda: {
            "executable": "tesseract",
            "version": "tesseract 5.5",
            "installed_languages": ["eng", "fra"],
        },
    )
    (models / "vision.onnx").write_bytes(content)
    return tmp_path / "workspace", models, selected


def test_ready_checks_preserve_source_and_use_selected_precision_without_network(
    device, monkeypatch
):
    workspace, models, selected = device
    source = workspace.parent / "photos"
    source.mkdir()
    original = source / "original.txt"
    original.write_bytes(b"leave me alone")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("Unexpected network"))
    progress = []
    report = readiness.inspect(
        workspace, models, siglip_precision="fp32", source=source, on_progress=progress.append
    )
    assert report["ready"] and report["languages"] == "eng+fra"
    assert selected == [("int8", "fp32")]
    assert len(report["checks"]) == 4 and all(check["ok"] for check in report["checks"])
    assert any(event.get("bytes_completed") for event in progress)
    assert progress[-1]["phase"] == "Setup check finished"
    assert original.read_bytes() == b"leave me alone" and list(workspace.iterdir()) == []


@pytest.mark.parametrize("problem", ["missing", "truncated", "corrupt"])
def test_missing_and_corrupt_model_files_are_not_ready(device, problem):
    workspace, models, selected = device
    path = models / "vision.onnx"
    if problem == "missing":
        path.unlink()
    elif problem == "truncated":
        path.write_bytes(b"short")
    else:
        path.write_bytes(b"x" * path.stat().st_size)
    report = readiness.inspect(workspace, models)
    assert not report["ready"]
    assert not next(check for check in report["checks"] if check["id"] == "models")["ok"]


def test_missing_requested_languages_are_not_silently_replaced(device):
    workspace, models, selected = device
    report = readiness.inspect(workspace, models, languages="eng+ara")
    assert not report["ready"] and report["languages"] == "eng+ara"
    assert "ara" in next(check for check in report["checks"] if check["id"] == "ocr")["detail"]
    assert report["ocr"]["installed_languages"] == ["eng", "fra"]


def test_permission_errors_and_unavailable_source_are_reported_individually(device, monkeypatch):
    workspace, models, selected = device

    def denied(*a, **k):
        raise PermissionError("workspace is read-only")

    monkeypatch.setattr(readiness.tempfile, "TemporaryFile", denied)
    report = readiness.inspect(workspace, models, source=workspace / "missing")
    checks = {check["id"]: check for check in report["checks"]}
    assert not checks["workspace"]["ok"] and not checks["source"]["ok"]
    assert checks["models"]["ok"] and checks["ocr"]["ok"]


def test_corrupt_workspace_is_an_actionable_failure(device):
    workspace, models, selected = device
    workspace.mkdir()
    (workspace / "catalog.sqlite3").write_bytes(b"not a database")
    report = readiness.inspect(workspace, models)
    assert not report["ready"] and not report["checks"][0]["ok"]


def test_cancellation_during_model_hashing_escapes_instead_of_reporting_readiness(device):
    workspace, models, selected = device

    def cancel(event):
        if event.get("current_file"):
            raise readiness.CheckCancelled()

    with pytest.raises(readiness.CheckCancelled):
        readiness.inspect(workspace, models, on_progress=cancel)


@pytest.mark.parametrize("case", ["missing", "old", "timeout", "no-packs"])
def test_ocr_diagnostics_are_bounded_and_report_missing_or_unusable_installations(
    tmp_path, monkeypatch, case
):
    monkeypatch.setattr(
        readiness.shutil, "which", lambda name: None if case == "missing" else "tesseract"
    )

    def run(command, **kwargs):
        assert kwargs["timeout"] == 10 and kwargs["encoding"] == "utf-8"
        if case == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        return SimpleNamespace(
            stdout=(
                ("tesseract 4.1\n" if case == "old" else "tesseract 5.5\n")
                if command[-1] == "--version"
                else "List of available languages (1):\nosd\n"
            )
        )

    monkeypatch.setattr(readiness.subprocess, "run", run)
    if case == "no-packs":
        assert readiness.installed_ocr()["installed_languages"] == []
    else:
        with pytest.raises((ValueError, subprocess.TimeoutExpired)):
            readiness.installed_ocr()


def test_ocr_language_detection_ignores_headers_and_orientation_data(monkeypatch):
    monkeypatch.setattr(readiness.shutil, "which", lambda name: "tesseract")
    monkeypatch.setattr(
        readiness.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout=(
                "tesseract 5.5\n"
                if command[-1] == "--version"
                else 'List of available languages in "/example" (4):\neng\nara\nosd\nchi_sim\n'
            )
        ),
    )
    assert readiness.installed_ocr()["installed_languages"] == ["ara", "chi_sim", "eng"]
    assert readiness.suggested_languages(["deu"]) == "deu"


@pytest.mark.parametrize("value", ["eng+--help", "eng\nfra", "../eng"])
def test_language_codes_are_validated_before_checking(device, value):
    with pytest.raises(ValueError, match="language codes"):
        readiness.inspect(device[0], device[1], languages=value)
