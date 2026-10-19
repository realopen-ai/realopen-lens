"""Regression cases for observed preprocessing, OCR, fusion, and resume failures."""

import pytest


from PIL import Image


from realopen_lens.inference import image_rgb


@pytest.mark.parametrize("mode", ["RGBA", "P"])
def test_transparent_screenshot_border_is_white_not_black(tmp_path, mode):
    # Rich's empty-OCR failure was caused by discarding transparent black pixels.
    path = tmp_path / "terminal.png"
    source = Image.new("RGBA", (12, 10), (0, 0, 0, 0))
    source.paste((35, 35, 35, 255), (3, 3, 9, 7))
    if mode == "P":
        source = source.quantize()
    source.save(path)
    original = path.read_bytes()
    with image_rgb(path) as actual:
        assert actual.mode == "RGB"
        assert actual.getpixel((0, 0)) == (255, 255, 255)
        assert actual.getpixel((5, 5)) == (35, 35, 35)
        assert not actual.info
    assert path.read_bytes() == original


def test_relative_corpus_paths_and_invalid_limits(tmp_path, monkeypatch):
    from realopen_lens.corpus import scan

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "good.png").touch()
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    assert scan(Path("images"), 1) == ["good.png"]
    with pytest.raises(ValueError, match="positive"):
        scan(Path("images"), 0)


def test_oversized_encoded_file_is_rejected_before_decode(tmp_path):
    path = tmp_path / "large.png"
    with path.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="encoded size"):
        image_rgb(path)


def test_unsupported_content_is_rejected_even_with_an_allowed_extension(tmp_path):
    path = tmp_path / "disguised.png"
    Image.new("RGB", (10, 10), "white").save(path, format="GIF")
    with pytest.raises(ValueError, match="Unsupported image content format"):
        image_rgb(path)
