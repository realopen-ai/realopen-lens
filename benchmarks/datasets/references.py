"""Validated benchmark manifests and reviewed reference corrections."""

import json
from pathlib import Path
from realopen_lens.models.manager import digest
from benchmarks.paths import SOURCES


def inputs(dataset: Path) -> list[dict]:
    rows = json.loads((dataset / "manifest.json").read_text())["images"]
    edits = {r["id"]: r for r in json.loads(CORRECTIONS.read_text())["images"]}
    historical = json.loads((dataset / "manifest.json").read_text())["version"] == 2
    for row in rows:
        if digest(dataset / row["path"]) != row["sha256"]:
            raise ValueError(f"Image changed: {row['id']}")
        if historical and row["id"] in edits:
            edit = edits[row["id"]]
            if row.get("key_phrases") != edit["expected_key_phrases"]:
                raise ValueError(f"Reference correction precondition failed: {row['id']}")
            row["key_phrases"] = edit["key_phrases"]
            row["evaluation_correction"] = edit
    return rows


CORRECTIONS = SOURCES / "evaluation-corrections-v1.json"
