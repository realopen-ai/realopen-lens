"""Validate optional OCR result conversion without installing models in normal CI."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from benchmarks.ocr_compare import record
from benchmarks.rapidocr_engine import RapidEngine, parse_output


def test_line_polygons_and_confidence_are_preserved_without_fabricating_word_boxes():
    result = parse_output(
        SimpleNamespace(
            txts=["README.md"], scores=[0.95], boxes=[[[1, 2], [10, 2], [10, 5], [1, 5]]]
        )
    )
    assert result["text"] == "README.md"
    assert result["lines"][0]["confidence"] == 95
    assert result["lines"][0]["polygon"][0] == [1, 2]
    assert "words" not in result
    assert parse_output(SimpleNamespace(txts=None, scores=None, boxes=None))["text"] == ""
    with pytest.raises(ValueError, match="lengths"):
        parse_output(SimpleNamespace(txts=["a"], scores=[], boxes=[]))
    with pytest.raises(ValueError, match="confidence"):
        parse_output(SimpleNamespace(txts=["a"], scores=[np.nan], boxes=[np.zeros((4, 2))]))


def test_warmup_recognition_only_does_not_disable_detection_on_real_images(tmp_path, monkeypatch):
    calls = []

    class FakeOCR:
        def __init__(self, params):
            pass

        def __call__(self, image, **kwargs):
            calls.append(kwargs)
            assert image.getpixel((0, 0)) == (255, 255, 255)
            return SimpleNamespace(txts=None, scores=None, boxes=None)

    module = ModuleType("rapidocr")
    module.RapidOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "rapidocr", module)
    names = [
        "PP-OCRv6_det_small.onnx",
        "PP-OCRv6_rec_small.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    ]
    monkeypatch.setattr(
        "benchmarks.rapidocr_engine.model_info",
        lambda: {"artifacts": {n: {"path": n} for n in names}},
    )
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(path)
    engine = RapidEngine()
    engine.extract(path)
    assert calls[-1] == {"use_det": True, "use_cls": True, "use_rec": True}


def test_execution_failure_cannot_be_scored_as_successful_empty_ocr():
    row = {"id": "x", "expected": {"decode": "success"}, "width": 100, "height": 100}
    result = record(row, {"status": "timeout", "error": "deadline"})
    assert result["quality_status"] == "fail"
    assert result["status"] == "timeout"
    assert result["responsible_stage"] == "ocr"
    row["expected"]["decode"] = "reject"
    result = record(row, {"status": "error", "error": "exceeds decoded pixels"})
    assert result["quality_status"] == "pass"
    assert result["status"] == "expected_rejection"


def test_worker_keeps_completed_heartbeat_between_items_and_can_resume(tmp_path, monkeypatch):
    import json
    from benchmarks.ocr_compare import worker
    from benchmarks.suite import connect

    monkeypatch.setattr("realopen_lens.offline.enforce", lambda: None)
    monkeypatch.setattr("realopen_lens.ocr._environment", lambda *args: {"version": "test"})
    monkeypatch.setattr(
        "realopen_lens.ocr.extract", lambda *args: {"text": "example", "warnings": []}
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset": str(tmp_path),
                "threads": 2,
                "languages": "eng",
                "images": [{"id": "a", "path": "a.png"}, {"id": "b", "path": "b.png"}],
            }
        )
    )
    worker(config, "tesseract")
    db = connect(tmp_path / "results.sqlite")
    heartbeat = json.loads(
        db.execute("SELECT value FROM metadata WHERE key='tesseract_current'").fetchone()[0]
    )
    assert heartbeat["id"] == "b"
    assert db.execute("SELECT COUNT(*) FROM stages").fetchone()[0] == 2
    worker(config, "tesseract")
    assert db.execute("SELECT COUNT(*) FROM stage_history").fetchone()[0] == 0
    db.close()
