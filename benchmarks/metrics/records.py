"""Per-image OCR comparison records with explicit quality verdicts."""

from benchmarks.metrics.quality import image_quality, quality_status


def record(row: dict, result: dict) -> dict:
    expected = row["expected"]["decode"] == "reject"
    rejected = expected and "decoded pixels" in result.get("error", "")
    quality = image_quality(row, result.get("text", "")) if result["status"] == "ok" else {}
    verdict = ("pass" if rejected else "fail") if expected else quality_status(quality)
    if result["status"] != "ok" and not rejected:
        verdict = "fail"
    status = "expected_rejection" if rejected else result["status"]
    if status == "ok" and (result.get("warnings") or verdict == "fail"):
        status = "degraded"
    pixels = row["width"] * row["height"]
    return {
        **row,
        "status": status,
        "quality_status": verdict,
        "quality": quality,
        "actual": result,
        "responsible_stage": "ocr" if verdict == "fail" else None,
        "latency_seconds": result.get("seconds", 0),
        "peak_tree_rss_mib": result.get("peak_tree_rss_mib"),
        "resolution": (
            "over-24MP"
            if pixels > 24_000_000
            else "6-24MP" if pixels > 6_000_000 else "1-6MP" if pixels > 1_000_000 else "under-1MP"
        ),
    }
