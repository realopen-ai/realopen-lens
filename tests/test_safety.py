import hashlib


import json


import pytest


from realopen_lens import models


def test_model_verification_detects_tampering_without_network(tmp_path, monkeypatch):
    content = b"pinned model"
    spec = {
        "path": "model.onnx",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    manifest = {"models": {"e5": {"revision": "abc", "files": {"text_int8": spec}}}}
    monkeypatch.setattr(models, "registry", lambda: manifest)
    path = tmp_path / "e5" / "abc" / "model.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    models.verify(tmp_path, "int8")
    path.write_bytes(b"broken model")
    with pytest.raises(ValueError, match="integrity"):
        models.verify(tmp_path, "int8")


def test_model_manifest_pins_full_revisions_and_sha256():
    for model in models.registry()["models"].values():
        assert len(model["revision"]) == 40
        for artifact in model["files"].values():
            assert len(artifact["sha256"]) == 64
            assert artifact["size"] > 0
    # Also catch accidentally truncated or malformed package data.
    json.dumps(models.registry())
