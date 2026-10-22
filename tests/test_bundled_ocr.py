"""Private runtime selection, isolation, integrity and packaging failures."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import pytest

from realopen_lens.ocr import runtime


def packaging_module():
    path = Path(__file__).resolve().parents[1] / "packaging/bundle_ocr.py"
    spec = importlib.util.spec_from_file_location("lens_bundle_ocr", path)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def private_runtime(tmp_path, monkeypatch):
    app = tmp_path / "Lens.app"
    engine = app / "Contents/Frameworks/lens-ocr/bin/tesseract"
    resources = app / "Contents/Resources/lens-ocr"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"private engine")
    (resources / "tessdata/configs").mkdir(parents=True)
    tsv = resources / "tessdata/configs/tsv"
    tsv.write_bytes(b"tessedit_create_tsv 1\n")
    manifest = {
        "version": 1,
        "engine": "tesseract 5",
        "files": [
            {
                "file": engine.relative_to(app).as_posix(),
                "sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            }
        ],
        "tessdata": {},
        "tsv_sha256": hashlib.sha256(tsv.read_bytes()).hexdigest(),
    }
    for name in ("eng", "fra", "ara", "osd"):
        data = name.encode()
        (resources / "tessdata" / f"{name}.traineddata").write_bytes(data)
        manifest["tessdata"][name] = {"sha256": hashlib.sha256(data).hexdigest()}
    (resources / "runtime.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runtime, "bundle_paths", lambda: (engine, resources))
    return engine, resources, manifest


def test_source_install_keeps_system_engine_and_environment(monkeypatch):
    monkeypatch.setattr(runtime, "bundle_paths", lambda: None)
    monkeypatch.setenv("TESSDATA_PREFIX", "user-selected-data")
    assert runtime.resolve("system-tesseract") == "system-tesseract"
    assert runtime.resolve(None) is None
    assert runtime.environment("system-tesseract")["TESSDATA_PREFIX"] == "user-selected-data"
    assert runtime.verify("system-tesseract") is None


def test_private_engine_ignores_host_prefix_and_loader_overrides(private_runtime, monkeypatch):
    engine, resources, _ = private_runtime
    monkeypatch.setenv("TESSDATA_PREFIX", "wrong-host-data")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "wrong-host-libraries")
    assert runtime.resolve("host-tesseract") == str(engine)
    assert runtime.resolve(None) == str(engine)
    environment = runtime.environment(str(engine))
    assert environment["TESSDATA_PREFIX"] == str(resources / "tessdata")
    assert "DYLD_LIBRARY_PATH" not in environment
    assert os.environ["TESSDATA_PREFIX"] == "wrong-host-data"
    assert runtime.verify(str(engine))["engine"] == "tesseract 5"
    assert runtime.fingerprint(str(engine)) is not None


@pytest.mark.parametrize("part", ["engine", "ara", "osd", "tsv"])
def test_damaged_private_resource_is_reported(private_runtime, part):
    engine, resources, _ = private_runtime
    path = (
        engine
        if part == "engine"
        else (
            resources / "tessdata/configs/tsv"
            if part == "tsv"
            else resources / "tessdata" / f"{part}.traineddata"
        )
    )
    path.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="integrity check failed"):
        runtime.verify(str(engine))


def test_missing_private_engine_does_not_fall_back_to_homebrew(private_runtime):
    engine, _, _ = private_runtime
    engine.unlink()
    with pytest.raises(ValueError, match="reinstall"):
        runtime.resolve("homebrew-tesseract")


def test_manifest_cannot_escape_bundle(private_runtime):
    engine, resources, manifest = private_runtime
    manifest["files"].append({"file": "../outside", "sha256": "ignored"})
    (resources / "runtime.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        runtime.verify(str(engine))


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
