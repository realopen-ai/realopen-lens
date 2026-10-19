"""OCR comparison reports and validated merges of retained engine runs."""

import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any
from benchmarks.storage.results import connect
from benchmarks.metrics.statistics import summarize
from realopen_lens.models.manager import digest
from benchmarks.metrics.records import record
from benchmarks.runtime.identity import ENGINES


def report(output: Path, config: dict) -> dict:
    db = connect(output / "results.sqlite")
    summary: dict[str, Any] = {
        "completed_at": datetime.now(UTC).isoformat(),
        "config": config,
        "engines": {},
    }
    all_records = {}
    for engine in config.get("engines", ENGINES[:2]):
        payloads = {
            key: json.loads(value)
            for key, value in db.execute(
                "SELECT image_id,payload FROM stages WHERE stage=?", (engine,)
            )
        }
        if set(payloads) != {r["id"] for r in config["images"]}:
            raise ValueError(f"Incomplete {engine} run; resume instead of silently skipping")
        records = [record(row, payloads[row["id"]]) for row in config["images"]]
        all_records[engine] = {r["id"]: r for r in records}
        (output / f"{engine}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        metadata = {
            k: json.loads(v)
            for k, v in db.execute(
                "SELECT key,value FROM metadata WHERE key LIKE ?", (engine + "_%",)
            )
        }
        summary["engines"][engine] = {
            "aggregate": summarize(records),
            "metadata": metadata,
            "by_source": {
                kind: summarize([r for r in records if r["source"]["kind"] == kind])
                for kind in ("real", "synthetic", "derived")
                if any(r["source"]["kind"] == kind for r in records)
            },
        }
    for before_engine, after_engine in combinations(all_records, 2):
        changes = []
        for key, before in all_records[before_engine].items():
            after = all_records[after_engine][key]
            changes.append(
                {
                    "id": key,
                    "source_kind": before["source"]["kind"],
                    "category": before["category"],
                    "before": before["quality_status"],
                    "after": after["quality_status"],
                    before_engine + "_quality": before["quality"],
                    after_engine + "_quality": after["quality"],
                    before_engine + "_seconds": before["latency_seconds"],
                    after_engine + "_seconds": after["latency_seconds"],
                }
            )
        content = "".join(json.dumps(r) + "\n" for r in changes)
        (output / f"changes-{before_engine}-{after_engine}.jsonl").write_text(content)
        if (before_engine, after_engine) == ("tesseract", "rapidocr"):
            (output / "changes.jsonl").write_text(content)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    db.close()
    return summary


def merge_runs(sources: list[Path], output: Path) -> dict:
    """Combine disjoint engine runs only when inputs and evaluation are identical."""
    if output.exists():
        raise ValueError("Merged output must be a new directory")
    components = [(path, json.loads((path / "config.json").read_text())) for path in sources]
    if not components:
        raise ValueError("No component runs supplied")
    reference = components[0][1]
    selected: dict[str, Path] = {}
    for path, config in components:
        for key in (
            "manifest_sha256",
            "corrections_sha256",
            "images",
            "threads",
            "languages",
            "packages",
            "host",
            "evaluation_sha256",
            "image_decode_sha256",
        ):
            if key not in config or key not in reference:
                raise ValueError(f"Component run lacks identity field: {key}")
            if config[key] != reference[key]:
                raise ValueError(f"Component runs differ: {key}")
        for engine in config["engines"]:
            if engine in selected:
                raise ValueError(f"Duplicate engine component: {engine}")
            selected[engine] = path
    config = {
        **reference,
        "engines": [e for e in ENGINES if e in selected],
        "component_runs": {
            e: {"path": str(path), "config_sha256": digest(path / "config.json")}
            for e, path in selected.items()
        },
    }
    config.pop("source_sha256", None)
    for _, component in components:
        for key in ("glm", "glm_options", "ollama_server_pids", "rapidocr"):
            if component.get(key) is not None:
                config[key] = component[key]
    # Validate complete image coverage before creating any merged artifact.
    expected = {r["id"] for r in config["images"]}
    for engine, path in selected.items():
        source = connect(path / "results.sqlite")
        found = {
            r[0] for r in source.execute("SELECT image_id FROM stages WHERE stage=?", (engine,))
        }
        source.close()
        if found != expected:
            raise ValueError(f"Incomplete component run: {engine}")
    output.mkdir(parents=True)
    destination = connect(output / "results.sqlite")
    for engine, path in selected.items():
        source = connect(path / "results.sqlite")
        with destination:
            destination.executemany(
                "INSERT INTO stages VALUES (?,?,?)",
                source.execute(
                    "SELECT image_id,stage,payload FROM stages WHERE stage=?", (engine,)
                ),
            )
            destination.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                source.execute("SELECT key,value FROM metadata WHERE key LIKE ?", (engine + "_%",)),
            )
        source.close()
    destination.close()
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    return report(output, config)
