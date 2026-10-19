"""Validate retrieval evaluation contracts without running models or downloading images."""

import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.suite import connect, identifier_preservation, report, save_stage


def test_identifiers_require_exact_case_and_punctuation_and_word_boundaries():
    identifiers = ["LENS-7300", "inventory_sync", "/api/v2/inventory/1"]
    assert all(identifier_preservation(identifiers, " ".join(identifiers)).values())
    assert not any(
        identifier_preservation(
            identifiers, "lens-7300 inventory-sync /api/v2/inventory/10"
        ).values()
    )


@pytest.mark.parametrize(
    "status,text,expected", [("ok", "", "pass"), ("ok", "noise", "fail"), ("error", "", "fail")]
)
def test_negative_empty_output_only_passes_when_ocr_succeeded(tmp_path, status, text, expected):
    row = {
        "id": "photo",
        "path": "photo.jpg",
        "width": 10,
        "height": 10,
        "category": "photos",
        "subcategory": "negative",
        "format": "JPEG",
        "split": "holdout",
        "difficulty": [],
        "expected": {"decode": "success"},
        "negative_text": True,
    }
    db = connect(tmp_path / "results.sqlite")
    for stage in ["ocr", "vision", "semantic"]:
        save_stage(db, "photo", stage, {"status": status if stage == "ocr" else "ok", "text": text})
    db.close()
    report(
        tmp_path, {"dataset": str(tmp_path), "images": [row], "queries": [], "variant": "improved"}
    )
    item = json.loads((tmp_path / "individual.jsonl").read_text())
    assert item["quality_status"] == expected


@pytest.mark.parametrize("gated", [False, True])
def test_worker_loads_detector_once_and_keeps_item_failures_durable(tmp_path, monkeypatch, gated):
    from benchmarks import suite_worker
    from realopen_lens import ocr, text_regions

    initialized, calls = [], []
    detector = SimpleNamespace()

    def create(path, threads):
        initialized.append((path, threads))
        return detector

    def extract(path, languages, threads, **options):
        calls.append(options)
        if path.name == "bad.png":
            raise ValueError("broken input")
        return {"text": "LENS-7300", "warnings": []}

    monkeypatch.setattr(text_regions, "TextRegionDetector", create)
    monkeypatch.setattr(ocr, "extract", extract)
    monkeypatch.setattr(suite_worker, "enforce", lambda: None)
    config = {
        "dataset": str(tmp_path),
        "models": str(tmp_path),
        "threads": 2,
        "languages": "eng",
        "variant": "improved",
        "queries": [],
        "images": [{"id": "a", "path": "bad.png"}, {"id": "b", "path": "good.png"}],
    }
    if gated:
        config["text_detector"] = {"path": str(tmp_path / "detector.onnx")}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["worker", str(path), "ocr"])
    suite_worker.main()
    assert len(initialized) == int(gated)
    assert len(calls) == 2
    assert all(
        c["detector"] is (detector if gated else None) and c["region_mode"] == "gate" for c in calls
    )
    db = connect(tmp_path / "results.sqlite")
    results = {
        key: json.loads(payload)
        for key, payload in db.execute("SELECT image_id,payload FROM stages")
    }
    assert results["a"]["status"] == "error"
    assert results["b"]["status"] == "ok"
    db.close()
