"""Private runtime selection, isolation, integrity and packaging failures."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


def packaging_module():
    path = Path(__file__).resolve().parents[1] / "packaging/bundle_ocr.py"
    spec = importlib.util.spec_from_file_location("lens_bundle_ocr", path)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("local_matches", [True, False])
def test_packaging_pins_local_and_downloaded_language_data(tmp_path, monkeypatch, local_matches):
    build = packaging_module()
    source = tmp_path / "source"
    source.mkdir()
    payload = b"verified language data"
    (source / "eng.traineddata").write_bytes(payload if local_matches else b"other version")
    downloads = []

    def fetch(url, **kwargs):
        downloads.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(build.urllib.request, "urlopen", fetch)
    registry = {
        "eng": {
            "url": "https://example.test/eng",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    }
    build.language_data(tmp_path / "data", source, registry)
    assert (tmp_path / "data/eng.traineddata").read_bytes() == payload
    assert bool(downloads) != local_matches
    assert (tmp_path / "data/configs/tsv").read_text() == "tessedit_create_tsv 1\n"


def test_packaging_rejects_bad_download(tmp_path, monkeypatch):
    build = packaging_module()
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad"))
    registry = {"eng": {"url": "https://example.test/eng", "size": 3, "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="Integrity check failed"):
        build.language_data(tmp_path / "data", tmp_path / "missing-source", registry)
    assert not (tmp_path / "data/eng.traineddata").exists()


def test_packaging_resolves_loader_relative_dependency_and_rejects_missing(tmp_path, monkeypatch):
    build = packaging_module()
    engine = tmp_path / "engine"
    engine.touch()
    library = tmp_path / "libexample.dylib"
    library.touch()

    def tool(*command):
        if "-l" in command:
            return "cmd LC_RPATH\ncmdsize 32\npath @loader_path (offset 12)"
        return (
            f"{engine}:\n  @rpath/libexample.dylib (compatibility version 1)\n"
            "  /usr/lib/libSystem.B.dylib (compatibility version 1)"
        )

    monkeypatch.setattr(build, "run", tool)
    assert build.dependencies(engine, engine) == [("@rpath/libexample.dylib", library)]
    library.unlink()
    with pytest.raises(ValueError, match="Cannot resolve"):
        build.dependencies(engine, engine)
