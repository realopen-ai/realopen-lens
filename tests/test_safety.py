import hashlib


import json


import pytest


from PIL import Image


from realopen_lens import models


from realopen_lens.inference import image_rgb


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


def test_model_manifest_pins_full_revisions_and_sha256():
    for model in models.registry()["models"].values():
        assert len(model["revision"]) == 40
        for artifact in model["files"].values():
            assert len(artifact["sha256"]) == 64
            assert artifact["size"] > 0
    # Also catch accidentally truncated or malformed package data.
    json.dumps(models.registry())
