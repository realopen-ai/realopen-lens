import hashlib
import json
import subprocess
import socket
import sys

import pytest
from PIL import Image

from realopen_lens.models import manager as models
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
    # The CLI installs an irreversible process-wide audit hook. Keep it out of
    # pytest: Windows asyncio uses a TCP loopback socketpair for its event loop.
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks", "run", "--data", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 1
    assert "Add images to data/" in result.stdout + result.stderr
    # Check TCP explicitly even on Unix, whose asyncio socketpair can hide this
    # leak. Later API tests must still be able to initialize an event loop.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2)
        with socket.create_connection(listener.getsockname(), timeout=2):
            connection, _ = listener.accept()
            connection.close()


def test_model_manifest_pins_full_revisions_and_sha256():
    for model in models.registry()["models"].values():
        assert len(model["revision"]) == 40
        for artifact in model["files"].values():
            assert len(artifact["sha256"]) == 64
            assert artifact["size"] > 0
    # Also catch accidentally truncated or malformed package data.
    json.dumps(models.registry())
