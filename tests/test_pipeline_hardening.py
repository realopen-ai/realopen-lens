"""Regression cases for observed preprocessing, OCR, fusion, and resume failures."""

import pytest


def test_relative_corpus_paths_and_invalid_limits(tmp_path, monkeypatch):
    from realopen_lens.corpus import scan

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "good.png").touch()
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    assert scan(Path("images"), 1) == ["good.png"]
    with pytest.raises(ValueError, match="positive"):
        scan(Path("images"), 0)
