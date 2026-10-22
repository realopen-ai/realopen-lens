"""Desktop startup boundaries must not compromise local worker isolation."""

import socket
import sys

import pytest

from realopen_lens.desktop import entry, launcher


def test_worker_dispatch_preserves_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["Lens", "-m", "realopen_lens.runtime.worker", "--query"])
    observed = []
    monkeypatch.setattr(entry.runpy, "run_module", lambda *a, **k: observed.append((a, k)))
    entry.main()
    assert observed == [(("realopen_lens.runtime.worker",), {"run_name": "__main__"})]
    assert sys.argv == ["realopen_lens.runtime.worker", "--query"]


@pytest.mark.parametrize("arguments", [["-m"], ["-m", "http.server"]])
def test_worker_dispatch_rejects_other_modules(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["Lens", *arguments])
    with pytest.raises(SystemExit, match="Unsupported"):
        entry.main()


def test_listener_refuses_occupied_port_without_attaching():
    with launcher.bind(0) as owned:
        port = owned.getsockname()[1]
        with pytest.raises(OSError):
            launcher.bind(port)
        assert owned.getsockname() == ("127.0.0.1", port)
    with launcher.bind(port) as released:
        assert released.family == socket.AF_INET


def test_models_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "workspace"
    assert launcher.model_root(root, {}) == root / "models"
    selected = tmp_path / "weights"
    assert launcher.model_root(root, {"models": str(selected)}) == selected


def test_settings_are_explicit_utf8_and_fail_cleanly(tmp_path):
    assert launcher.settings(tmp_path) == {}
    import json

    folder = str(tmp_path / "é")
    (tmp_path / "desktop.json").write_text(json.dumps({"roots": [folder]}), encoding="utf-8")
    assert launcher.settings(tmp_path)["roots"] == [folder]  # Legacy entries remain readable.
    (tmp_path / "desktop.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.settings(tmp_path)
