"""Packaging failures retain the evidence needed by the CI artifact upload."""

import json
from pathlib import Path
import subprocess
import sys


def test_model_free_packaging_failure_writes_report(tmp_path):
    script = Path(__file__).resolve().parents[1] / "packaging/ci_smoke.py"
    report = tmp_path / "diagnostics" / "report.json"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "missing.zip"), "--report", str(report)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["checks"][0]["name"] == "extract archive"
    assert saved["checks"][0]["status"] == "failed"
    assert saved["checks"][0]["error"]
