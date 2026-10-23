"""Only verified OCR assets may populate or satisfy the shared build cache."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def cache_builder(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "packaging"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "lens_prepare_tessdata", directory / "prepare_tessdata.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(tmp_path / "prepare_tessdata.py")
    data = {"eng": b"english fixture", "fra": b"french fixture"}
    registry = {
        name: {
            "url": f"https://example.test/{name}",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in data.items()
    }
    (tmp_path / "tessdata.json").write_text(json.dumps(registry), encoding="utf-8")
    calls = []

    def download(url, **kwargs):
        calls.append(url.rsplit("/", 1)[-1])
        return io.BytesIO(data[calls[-1]])

    import bundle_ocr

    monkeypatch.setattr(bundle_ocr.urllib.request, "urlopen", download)
    return module, tmp_path / "cache", data, calls


def test_cold_cache_downloads_once_and_warm_cache_verifies_without_downloads(cache_builder):
    module, cache, data, calls = cache_builder
    module.prepare(cache)
    assert calls == ["eng", "fra"]
    assert all(
        (cache / f"{name}.traineddata").read_bytes() == value for name, value in data.items()
    )
    module.prepare(cache)
    assert calls == ["eng", "fra"]


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_cache_hit_never_bypasses_integrity_verification(cache_builder, damage):
    module, cache, data, calls = cache_builder
    module.prepare(cache)
    eng = cache / "eng.traineddata"
    if damage == "missing":
        eng.unlink()
    else:
        eng.write_bytes(b"corruption")
    module.prepare(cache)
    assert calls == ["eng", "fra", "eng"]
    assert eng.read_bytes() == data["eng"]


def test_failed_download_never_replaces_existing_cached_files(cache_builder):
    module, cache, data, _ = cache_builder
    module.prepare(cache)
    eng = cache / "eng.traineddata"
    eng.unlink()
    data["eng"] = b"untrusted response"
    with pytest.raises(ValueError, match="Integrity check failed"):
        module.prepare(cache)
    assert not eng.exists()
    assert (cache / "fra.traineddata").read_bytes() == data["fra"]
    assert not list(cache.glob(".tessdata-*"))
