import hashlib
import json
import subprocess
import sys

import pytest
from PIL import Image
from typer.testing import CliRunner

from realopen_lens.models import manager as models
from benchmarks.__main__ import app
from realopen_lens.encoders.onnx import image_rgb


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


def test_worker_offline_policy_blocks_python_network_operations():
    code = """
from realopen_lens.runtime.offline import enforce
import socket
enforce()
try:
    socket.getaddrinfo('example.com', 443)
except PermissionError:
    print('blocked')
else:
    raise SystemExit('Network guard failed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "blocked"


def test_images_are_bounded_before_decode(tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (20, 20)).save(path)
    with pytest.raises(ValueError, match="decoded pixels"):
        image_rgb(path, max_pixels=100)


def test_exif_orientation_is_applied_without_modifying_original(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (20, 10))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    before = path.read_bytes()
    with image_rgb(path) as decoded:
        assert decoded.size == (10, 20)
    assert path.read_bytes() == before


def test_cli_does_not_download_when_corpus_empty(tmp_path):
    result = CliRunner().invoke(app, ["run", "--data", str(tmp_path)])
    assert result.exit_code == 1
    assert "Add images to data/" in result.output


def test_model_manifest_pins_full_revisions_and_sha256():
    for model in models.registry()["models"].values():
        assert len(model["revision"]) == 40
        for artifact in model["files"].values():
            assert len(artifact["sha256"]) == 64
            assert artifact["size"] > 0
    # Also catch accidentally truncated or malformed package data.
    json.dumps(models.registry())
