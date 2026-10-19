import hashlib
from collections import Counter
from importlib.resources import files
from pathlib import PurePosixPath

import pytest

from benchmarks.datasets.ocr import SPECS, ensure_download, manifest, publish
from benchmarks.metrics.ocr import normalized


def test_fixture_manifest_covers_three_categories_and_reviewed_references():
    entries = manifest()
    assert Counter(item["category"] for item in entries) == {
        "terminal": 3,
        "document": 3,
        "diagram": 3,
    }
    assert {item["image_id"] for item in entries} == set(SPECS)
    for item in entries:
        assert len(item["sha256"]) == 64
        assert item["url"].startswith("https://")
        assert ".." not in PurePosixPath(item["image_id"]).parts
        assert len(SPECS[item["image_id"]]["queries"]) == 3
        if "reference_resource" in item:
            reference = (
                files("benchmarks").joinpath("ocr_assets/" + item["reference_resource"]).read_text()
            )
            assert all(
                normalized(phrase) in normalized(reference)
                for phrase in SPECS[item["image_id"]]["key_phrases"]
            )
        else:
            assert len(item["reference_sha256"]) == 64


def test_verified_assets_are_reused_and_modified_assets_rejected(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"pinned asset")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    ensure_download("https://example.invalid/image.png", path, expected)
    path.write_bytes(b"edited")
    with pytest.raises(ValueError, match="differs from pinned"):
        ensure_download("https://example.invalid/image.png", path, expected)
    assert path.read_bytes() == b"edited"


def test_edited_annotations_are_not_overwritten(tmp_path):
    path = tmp_path / "queries.json"
    publish(path, [{"text": "hello"}])
    publish(path, [{"text": "hello"}])
    with pytest.raises(ValueError, match="edited annotations"):
        publish(path, [{"text": "different"}])
