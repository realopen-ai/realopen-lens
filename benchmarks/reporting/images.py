"""Per-image benchmark evidence, failure records, and aggregate reports."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from benchmarks.storage.results import connect
from benchmarks.metrics.statistics import summarize
from benchmarks.metrics.quality import image_quality, quality_status, identifier_preservation
from benchmarks.reporting.retrieval import retrieval_report


def report(output: Path, config: dict) -> dict:
    db = connect(output / "results.sqlite")
    rows = config["images"]
    records: list[dict[str, Any]] = []
    stage_lookup = {
        (key, stage): json.loads(payload)
        for key, stage, payload in db.execute("SELECT image_id,stage,payload FROM stages")
    }
    for row in rows:
        stages = {
            stage: stage_lookup.get((row["id"], stage), {"status": "missing"})
            for stage in ("ocr", "vision", "semantic")
        }
        statuses = [r["status"] for r in stages.values()]
        expected_reject = row["expected"]["decode"] == "reject"
        failures = [s for s, p in stages.items() if p["status"] in ("error", "timeout", "missing")]
        rejected = expected_reject and "decoded pixels" in stages["vision"].get("error", "")
        status = (
            "expected_rejection"
            if rejected
            else (
                "timeout"
                if "timeout" in statuses
                else (
                    "error"
                    if failures
                    else "degraded" if any(p.get("warnings") for p in stages.values()) else "ok"
                )
            )
        )
        quality = (
            image_quality(row, stages["ocr"].get("text", ""))
            if stages["ocr"]["status"] == "ok"
            else {}
        )
        quality_verdict = (
            ("pass" if rejected else "fail") if expected_reject else quality_status(quality)
        )
        if row.get("negative_text"):
            quality_verdict = (
                "pass"
                if stages["ocr"]["status"] == "ok" and not stages["ocr"].get("text", "").strip()
                else "fail"
            )
        if quality_verdict == "fail" and status == "ok":
            status = "degraded"
        if failures and not rejected:
            quality_verdict = "fail"
        pixels = row["width"] * row["height"]
        record = {
            **row,
            "status": status,
            "quality_status": quality_verdict,
            "quality": quality,
            "identifiers": identifier_preservation(
                row.get("identifiers", []), stages["ocr"].get("text", "")
            ),
            "stages": stages,
            "responsible_stages": failures or (["ocr"] if quality_verdict == "fail" else []),
            "resolution": (
                "over-24MP"
                if pixels > 24_000_000
                else (
                    "6-24MP"
                    if pixels > 6_000_000
                    else "1-6MP" if pixels > 1_000_000 else "under-1MP"
                )
            ),
            "latency_seconds": sum(p.get("seconds", 0) for p in stages.values()),
            "peak_tree_rss_mib": max(
                (p.get("peak_tree_rss_mib") or 0 for p in stages.values()), default=0
            ),
        }
        records.append(record)
    with (output / "individual.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    retrieval = retrieval_report(db, config, output)
    summary: dict[str, Any] = {
        "config": config,
        "completed_at": datetime.now(UTC).isoformat(),
        "images": len(rows),
        "aggregate": summarize(records),
        "retrieval": retrieval,
        "limitations": [
            "OCR diagnostics require >=80% token recall/precision and key-phrase recall; "
            "CER/WER remain order-sensitive diagnostics",
            "Untranscribed images have no quality verdict; execution is not accuracy",
            "RSS includes resident models and children; sampled, not incremental",
            "Stage sum latency excludes model loading and retrieval; consult wall times",
            "Relevance judgments are known positives and incomplete",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    failure_records = [
        {
            "id": r["id"],
            "category": r["category"],
            "split": r["split"],
            "status": r["status"],
            "responsible_stages": r["responsible_stages"],
            "quality": r["quality"],
            "latency_seconds": r["latency_seconds"],
            "image": str(Path(config["dataset"]) / r["path"]),
            "evidence": "individual.jsonl and results.sqlite: OCR, boxes, confidence, errors",
        }
        for r in records
        if r["status"] != "ok" or r["quality_status"] == "fail"
    ]
    (output / "failures.json").write_text(
        json.dumps(failure_records, ensure_ascii=False, indent=2) + "\n"
    )
    db.close()
    print(json.dumps(summary["aggregate"]["all"], indent=2), flush=True)
    return summary
