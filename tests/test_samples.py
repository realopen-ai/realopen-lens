import io
import json

import pytest
from PIL import Image

from benchmarks.datasets import photos as samples


def rows():
    return [
        {
            "filename": f"COCO_val2014_{i:012}.jpg",
            "filepath": "val2014",
            "split": "test",
            "cocoid": i,
            "sentences": [{"raw": f"A sample photograph {i}."}],
        }
        for i in range(20)
    ]


def test_sample_is_seeded_unique_and_excludes_invalid_paths():
    source = rows() + [{**rows()[0], "filename": "../../private.jpg"}]
    selected = samples.select(source, 5, 42)
    assert selected == samples.select(source, 5, 42)
    assert selected != samples.select(source, 5, 43)
    assert len({item["filename"] for item in selected}) == 5
    with pytest.raises(ValueError, match="Count"):
        samples.select(source, 501, 42)


def test_invalid_download_does_not_publish_an_image(tmp_path, monkeypatch):
    monkeypatch.setattr(samples, "get_bytes", lambda *args: b"not an image")
    with pytest.raises(OSError):
        samples.download_image(rows()[0], tmp_path)
    assert not list(tmp_path.iterdir())


def test_resume_verifies_images_and_preserves_edited_queries(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(buffer, format="JPEG")
    payload = buffer.getvalue()
    calls = []

    def get(url, limit):
        calls.append(url)
        return payload

    monkeypatch.setattr(samples, "annotations", lambda cache: rows())
    monkeypatch.setattr(samples, "get_bytes", get)
    output = tmp_path / "sample"
    query_file = samples.fetch(output, tmp_path / "cache", 3, 42, 1)
    assert len(calls) == 3
    assert len(json.loads(query_file.read_text())) == 3
    samples.fetch(output, tmp_path / "cache", 3, 42, 1)
    assert len(calls) == 3  # Verified cached images cause no further downloads.
    query_file.write_text("[]")
    with pytest.raises(ValueError, match="edited query"):
        samples.fetch(output, tmp_path / "cache", 3, 42, 1)
    assert query_file.read_text() == "[]"


def test_downloader_never_overwrites_existing_user_image(tmp_path, monkeypatch):
    row = rows()[0]
    path = tmp_path / row["filename"]
    path.write_bytes(b"user image")
    with pytest.raises(ValueError, match="overwrite"):
        samples.download_image(row, tmp_path)
    assert path.read_bytes() == b"user image"


def test_invalid_count_is_rejected_before_network_access(tmp_path, monkeypatch):
    def unexpected_network(cache):
        raise AssertionError("Invalid arguments must not trigger downloads")

    monkeypatch.setattr(samples, "annotations", unexpected_network)
    with pytest.raises(ValueError, match="Count"):
        samples.fetch(tmp_path / "out", tmp_path / "cache", 0, 42, 1)
