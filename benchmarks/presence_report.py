"""Compare paired retrieval runs with explicit negative and exact-identifier diagnostics."""

import argparse
import json
from pathlib import Path

from benchmarks.compare import compare, read_records
from benchmarks.suite import distribution


def evaluate(
    whole: Path, gate: Path, output: Path, source_change_reason: str | None = None
) -> dict:
    configs = [json.loads((root / "config.json").read_text()) for root in (whole, gate)]
    for key in ("engine_sha256", "runner_sha256", "worker_limits", "variant", "images"):
        if key in ("engine_sha256", "runner_sha256") and source_change_reason:
            continue
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Paired run identity differs: {key}")
    if configs[0].get("text_detector") or configs[1].get("text_detector", {}).get("mode") != "gate":
        raise ValueError("Expected whole-image versus explicitly gated current Tesseract")
    paired = compare(whole, gate, output)
    records = [read_records(root / "individual.jsonl") for root in (whole, gate)]
    queries = [read_records(root / "queries.jsonl") for root in (whole, gate)]
    for items, config in zip(records, configs, strict=True):
        if set(items) != {r["id"] for r in config["images"]}:
            raise ValueError("Image results are incomplete")
    variants = {}
    for name, root, items, query_rows in zip(
        ("whole", "gate"), (whole, gate), records, queries, strict=True
    ):
        strata = {}
        groups = {"all": list(items.values())}
        for corpus in ("historical", "fresh"):
            groups[corpus] = [r for r in items.values() if r["corpus"] == corpus]
        groups["fresh-positive"] = [r for r in groups["fresh"] if r.get("truth")]
        groups["fresh-negative"] = [r for r in groups["fresh"] if r.get("negative_text")]
        for label, rows in groups.items():
            negative = [r for r in rows if r.get("negative_text")]
            positive = [r for r in rows if r.get("truth") or r.get("key_phrases")]
            identifiers = [matched for r in rows for matched in r.get("identifiers", {}).values()]
            strata[label] = {
                "images": len(rows),
                "referenced_positive": len(positive),
                "positive_passed": sum(r["quality_status"] == "pass" for r in positive),
                "negative_images": len(negative),
                "negative_empty": sum(r["quality_status"] == "pass" for r in negative),
                "identifiers": len(identifiers),
                "identifiers_exact": sum(identifiers),
                "execution_errors": sum(r["status"] in ("error", "timeout") for r in rows),
                "expected_rejections": sum(r["status"] == "expected_rejection" for r in rows),
                "inference_skipped": sum(
                    bool(r["stages"]["ocr"].get("inference_skipped")) for r in rows
                ),
                "stage_sum_seconds": distribution([r["latency_seconds"] for r in rows]),
                "ocr_seconds": distribution([r["stages"]["ocr"].get("seconds", 0) for r in rows]),
                "rss_mib": distribution([r["peak_tree_rss_mib"] for r in rows]),
            }
        summary = json.loads((root / "summary.json").read_text())
        variants[name] = {
            "strata": strata,
            "retrieval": summary["retrieval"],
            "resources": json.loads((root / "resources.json").read_text()),
            "query_errors": [q["id"] for q in query_rows.values() if q["status"] != "ok"],
        }
    changes = []
    for key, new in records[1].items():
        old = records[0][key]
        old_text, new_text = (r["stages"]["ocr"].get("text", "") for r in (old, new))
        if old_text != new_text or (new["corpus"] == "fresh" and new["quality_status"] == "fail"):
            changes.append(
                {
                    "id": key,
                    "corpus": new["corpus"],
                    "difficulty": new["difficulty"],
                    "negative_text": new.get("negative_text", False),
                    "truth": new.get("truth"),
                    "whole": old_text,
                    "gate": new_text,
                    "whole_quality": old["quality"],
                    "gate_quality": new["quality"],
                    "whole_identifiers": old.get("identifiers", {}),
                    "gate_identifiers": new.get("identifiers", {}),
                    "gate_detection": new["stages"]["ocr"].get("text_detection"),
                }
            )
    misses = []
    for key, new in queries[1].items():
        old = queries[0][key]
        if (
            new["metrics"]["hybrid"]["hit_rate_at_5"] == 0
            or new["metrics"]["hybrid"] != old["metrics"]["hybrid"]
        ):
            misses.append(
                {
                    "id": key,
                    "text": new["text"],
                    "corpus": new.get("corpus"),
                    "relevant": new["relevant"],
                    "whole": old["metrics"],
                    "gate": new["metrics"],
                    "whole_rankings": old["rankings"],
                    "gate_rankings": new["rankings"],
                }
            )
    result = {
        "dataset_sha256": paired["dataset_sha256"],
        "engine_sha256": configs[0]["engine_sha256"],
        "harness_sha256": configs[0]["runner_sha256"],
        "gate_engine_sha256": configs[1]["engine_sha256"],
        "gate_harness_sha256": configs[1]["runner_sha256"],
        "source_change_reason": source_change_reason,
        "images": len(records[0]),
        "queries": len(queries[0]),
        "variants": variants,
        "ocr_transcripts_changed": sum(r["whole"] != r["gate"] for r in changes),
        "image_diagnostics": changes,
        "retrieval_diagnostics": misses,
        "hybrid_ranking_changes": sum(
            queries[0][k]["rankings"]["hybrid"] != q["rankings"]["hybrid"]
            for k, q in queries[1].items()
        ),
        "limitations": [
            (
                "Photo text presence was reviewed by one assistant "
                "before inference, not independently by humans."
            ),
            (
                "Authored screenshots supply exact ground truth but are "
                "synthetic and share six layout families."
            ),
            (
                "Fresh cases were frozen before inference; this is a "
                "diagnostic addition, not an independent blind study."
            ),
            "Known-positive retrieval labels are incomplete; unknown OCR accuracy remains unknown.",
            "Timing is sequential on one host; sampled process-tree RSS is not system RAM.",
        ],
    }
    (output / "presence-comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-change-reason",
        help="Explicitly document a reviewed source change when comparing a measured fix",
    )
    args = parser.parse_args()
    result = evaluate(args.whole, args.gate, args.output, args.source_change_reason)
    print(
        json.dumps(
            {
                "images": result["images"],
                "queries": result["queries"],
                "changed_transcripts": result["ocr_transcripts_changed"],
            }
        )
    )


if __name__ == "__main__":
    main()
