"""Pinned model downloads report progress and clean up cancelled artifacts."""

import pytest


def test_download_progress_and_cancellation_remove_unpublished_file(tmp_path, monkeypatch):
    import hashlib
    import io
    from realopen_lens.models import manager

    payload = b"model"
    spec = {
        "path": "model.onnx",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setattr(manager, "artifacts", lambda *a: [("demo", "vision_int8", spec)])
    monkeypatch.setattr(
        manager,
        "registry",
        lambda: {"models": {"demo": {"repo": "demo/demo", "revision": "pinned"}}},
    )
    destination = tmp_path / "model.onnx"
    monkeypatch.setattr(manager, "artifact_path", lambda *a: destination)
    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))

    def cancel(name, received, total):
        assert (name, received, total) == ("demo/model.onnx", len(payload), len(payload))
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        manager.fetch(tmp_path, "int8", lambda *a: None, download_progress=cancel)
    assert not destination.exists()
    assert not destination.with_suffix(".onnx.part").exists()
    observed = []
    manager.fetch(
        tmp_path, "int8", lambda *a: None, download_progress=lambda *a: observed.append(a)
    )
    assert observed == [("demo/model.onnx", len(payload), len(payload))]
    assert destination.read_bytes() == payload
