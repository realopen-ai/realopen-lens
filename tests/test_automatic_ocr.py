"""Script selection is bounded, auditable and independent of ground truth labels."""

import subprocess
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from test_collections import app as _app, image
from realopen_lens.ocr import automatic, tesseract
from realopen_lens.runtime import readiness

app = _app


def probe(script, confidence=5, region="full", status="ok"):
    return {"script": script, "confidence": confidence, "region": region, "status": status}


@pytest.mark.parametrize(
    "script,expected",
    [
        ("Latin", "fra"),
        ("Arabic", "urd"),
        ("Devanagari", "hin"),
        ("Bengali", "ben"),
        ("Han", "chi_tra"),
        ("Japanese", "jpn"),
        ("Korean", "kor"),
        ("Cyrillic", "rus"),
    ],
)
def test_confident_groups_select_corresponding_packs(script, expected):
    result = automatic.choose([probe(script)], list(automatic.LANGUAGES))
    assert expected in result["selected_languages"].split("+")
    assert "eng" in result["selected_languages"].split("+")
    assert result["fallback"] is None
    assert len(result["selected_languages"].split("+")) < 19


def test_weak_unknown_or_missing_group_falls_back_without_claiming_certainty():
    installed = ["eng", "fra", "ara"]
    for probes in [[probe("Latin", 0.5)], [probe("Greek")], [probe("Devanagari")], []]:
        result = automatic.choose(probes, installed)
        assert result["fallback"]
        assert result["selected_languages"] == "eng+fra+ara"
    with pytest.raises(ValueError):
        automatic.choose([], [])


def test_mixed_strong_probes_keep_both_groups_and_two_agreeing_bands_can_resolve():
    result = automatic.choose(
        [probe("Latin"), probe("Arabic", region="bottom")], list(automatic.LANGUAGES)
    )
    assert result["groups"] == ["Latin", "Arabic"]
    assert {"fra", "ara", "urd"}.issubset(result["selected_languages"].split("+"))
    result = automatic.choose(
        [probe("Korean", 0.5), probe("Korean", region="top"), probe("Korean", region="bottom")],
        ["kor", "eng"],
    )
    assert result["selected_languages"] == "kor+eng" and result["fallback"] is None


def test_detector_timeouts_are_bounded_and_produce_explicit_fallback(monkeypatch):
    calls = []

    def timeout(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(automatic.subprocess, "run", timeout)
    with Image.new("RGB", (800, 600)) as source:
        result = automatic.detect(source, "tesseract", ["eng", "fra"], 2, 1)
    assert len(calls) == 3 and all(0 < value <= 1 for value in calls)
    assert result["fallback"] == "uncertain_script"
    assert all(row["status"] == "timeout" for row in result["probes"])


def test_auto_environment_requires_osd_and_binds_policy_and_candidate_pack_hashes(
    tmp_path, monkeypatch
):
    for code in ("eng", "hin", "osd"):
        (tmp_path / f"{code}.traineddata").write_bytes(code.encode())
    osd = True

    def run(command, **kwargs):
        if command[-1] == "--version":
            return SimpleNamespace(stdout="tesseract 5\n")
        return SimpleNamespace(
            stdout=f'List of available languages in "{tmp_path}" (3):\neng\nhin\n'
            + ("osd\n" if osd else "")
        )

    monkeypatch.setattr(tesseract.subprocess, "run", run)
    monkeypatch.setattr(tesseract.runtime, "bundle_paths", lambda: None)
    tesseract._environment.cache_clear()
    try:
        result = tesseract._environment("fixture", "auto")
        assert set(result["language_pack_sha256"]) == {"eng", "hin"}
        assert len(result["automatic_language_policy"]["source_sha256"]) == 64
        osd = False
        tesseract._environment.cache_clear()
        with pytest.raises(ValueError, match="osd"):
            tesseract._environment("fixture", "auto")
    finally:
        tesseract._environment.cache_clear()


def test_manual_to_auto_reuses_metadata_and_vision(app):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    calls.clear()
    result = run(languages="auto")
    assert result.reused_stages == {"metadata": 1, "vision": 1, "ocr": 0, "semantic": 0}


def test_auto_readiness_reports_missing_orientation_and_usable_groups(tmp_path, monkeypatch):
    def installed():
        return {
            "installed_languages": ["eng", "hin"],
            "version": "tesseract 5",
            "orientation_available": True,
        }

    monkeypatch.setattr(readiness, "installed_ocr", installed)
    report = readiness.inspect(tmp_path / "workspace", tmp_path / "models", languages="auto")
    assert report["languages"] == "auto"
    assert next(c for c in report["checks"] if c["id"] == "ocr")["ok"]
    monkeypatch.setattr(
        readiness, "installed_ocr", lambda: {**installed(), "orientation_available": False}
    )
    report = readiness.inspect(tmp_path / "workspace", tmp_path / "models", languages="auto")
    assert not next(c for c in report["checks"] if c["id"] == "ocr")["ok"]


def test_policy_identity_survives_persistence_and_unchanged_auto_reuses_all_stages(
    app, monkeypatch
):
    from realopen_lens.indexing import service

    assert automatic.policy() == json.loads(json.dumps(automatic.policy()))
    root, store, calls, run = app
    monkeypatch.setattr(service, "verify", lambda *a: None)
    monkeypatch.setattr(
        tesseract, "_environment", lambda *a: {"automatic_language_policy": automatic.policy()}
    )
    image(root / "one.png")
    run(languages="auto")
    calls.clear()
    result = run(languages="auto")
    assert not calls
    assert result.reused_stages == {"metadata": 1, "vision": 1, "ocr": 1, "semantic": 1}


def test_published_routing_survives_later_mutable_analysis_changes(app, monkeypatch):
    from realopen_lens.api.viewer import details
    from realopen_lens.storage.catalog import Catalog
    from realopen_lens.storage.artifacts import load_manifest
    from realopen_lens.ocr import tesseract

    root, store, calls, run = app
    image(root / "one.png")
    original = tesseract.extract
    monkeypatch.setattr(
        tesseract,
        "extract",
        lambda *a, **k: {
            **original(*a, **k),
            "ocr_language_mode": "auto",
            "selected_languages": "eng+fra",
            "script_detection": {"groups": ["Latin"], "fallback": None},
        },
    )
    indexed = run(languages="auto")
    catalog = Catalog(store)
    try:
        _, active = catalog.snapshot("personal")
        manifest = load_manifest(catalog, active)
        key = manifest["images"][0]["image_id"]
        row = catalog.analysis(manifest["pipeline"], key, "ocr")
        payload = json.loads(row["payload"])
        catalog.save_analysis(
            manifest["pipeline"], key, "ocr", {**payload, "selected_languages": "jpn+eng"}
        )
    finally:
        catalog.close()
    assert (
        details(store, "personal", key, indexed.generation)["ocr_language_selection"][
            "selected_languages"
        ]
        == "eng+fra"
    )


def test_latin_route_with_only_nonlatin_packs_falls_back_instead_of_selecting_empty():
    selected = automatic.choose([probe("Latin")], ["kor"])
    assert selected["selected_languages"] == "kor"
    assert selected["fallback"] == "missing_group_packs"
