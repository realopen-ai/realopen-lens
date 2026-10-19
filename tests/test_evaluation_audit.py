"""Reference mistakes must be corrected explicitly, without altering retained inference."""

import json

import pytest

from benchmarks.datasets.metadata import annotation_matches
from benchmarks.reporting.rescore import rescore


def test_annotations_require_repository_and_exact_source_path():
    annotation = {"repo": "streamlit/docs", "source_path": "images/header.png"}
    assert annotation_matches(
        {"source": {"repo": "streamlit/docs", "path": "images/header.png"}}, annotation
    )
    assert not annotation_matches(
        {"source": {"repo": "mermaid-js/mermaid", "path": "docs/public/header.png"}}, annotation
    )
    assert not annotation_matches(
        {"source": {"repo": "streamlit/docs", "path": "other/header.png"}}, annotation
    )


def test_audited_rescore_preserves_outputs_and_checks_old_reference(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    row = {
        "id": "a",
        "key_phrases": ["wrong"],
        "category": "diagrams",
        "split": "holdout",
        "format": "PNG",
        "subcategory": "flow",
        "difficulty": [],
        "resolution": "under-1MP",
        "status": "ok",
        "quality_status": "fail",
        "quality": {},
        "latency_seconds": 1,
        "peak_tree_rss_mib": 10,
        "stages": {"ocr": {"text": "actual", "status": "ok"}},
    }
    (source / "individual.jsonl").write_text(json.dumps(row) + "\n")
    (source / "queries.jsonl").write_text("")
    (source / "summary.json").write_text(json.dumps({"config": {"queries": []}}))
    correction = tmp_path / "corrections.json"
    protocol = {
        "images": [{"id": "a", "expected_key_phrases": ["wrong"], "key_phrases": ["actual"]}],
        "exclude_queries": [],
    }
    correction.write_text(json.dumps(protocol))
    rescore(source, tmp_path / "audited", correction)
    corrected = json.loads((tmp_path / "audited/individual.jsonl").read_text())
    assert corrected["quality_status"] == "pass"
    assert corrected["stages"] == row["stages"]
    assert json.loads((source / "individual.jsonl").read_text()) == row
    protocol["images"][0]["expected_key_phrases"] = ["different"]
    correction.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="precondition"):
        rescore(source, tmp_path / "bad", correction)
    with pytest.raises(ValueError, match="new output"):
        rescore(source, source, correction)

    protocol["images"][0]["expected_key_phrases"] = ["wrong"]
    correction.write_text(json.dumps(protocol))
    row["stages"]["ocr"]["status"] = "error"
    (source / "individual.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="failed OCR"):
        rescore(source, tmp_path / "failed-ocr", correction)
