"""Published command compatibility and source-identity contracts."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "command",
    [
        "dataset",
        "samples",
        "ocr_samples",
        "presence_dataset",
        "text_region_controls",
        "probes",
        "suite",
        "compare",
        "calibrate",
        "ocr_compare",
        "text_regions",
        "presence_report",
        "rescore",
        "datasets.encoder_probes",
        "datasets.encoder_references",
        "datasets.ocr_probes",
        "experiments.candidate_depths",
        "experiments.encoder_parity",
        "experiments.precision_retrieval",
        "experiments.query_expansion",
        "experiments.arabic_lexical",
        "experiments.ocr_recovery",
        "experiments.siglip_precision",
    ],
)
def test_published_commands_still_accept_help(command):
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks." + command, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout


def test_source_identity_includes_relative_paths_and_contents(tmp_path):
    from benchmarks.runtime.identity import source_hash

    path = tmp_path / "module.py"
    path.write_text("value = 1\n")
    original = source_hash(tmp_path)
    path = path.rename(tmp_path / "renamed.py")
    renamed = source_hash(tmp_path)
    assert renamed != original
    path.write_text("value = 2\n")
    assert source_hash(tmp_path) != renamed


def test_baseline_loader_retains_its_worker_entry_point(tmp_path, monkeypatch):
    from benchmarks.runtime.suite_worker import baseline_ocr
    from realopen_lens.ocr import tesseract

    def extract(*args):
        return {"version": "fixture", "language_pack_sha256": {}, "text": "ERR_42"}

    monkeypatch.setattr(tesseract, "extract", extract)
    result = baseline_ocr(
        {
            "engine_root": str(Path(__file__).resolve().parents[1] / "src"),
            "dataset": str(tmp_path),
            "threads": 1,
            "languages": "eng",
        },
        {"path": "example.png"},
    )
    assert result["text"] == "ERR_42"
