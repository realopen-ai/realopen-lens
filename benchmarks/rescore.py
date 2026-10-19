"""Apply explicit reference corrections to retained outputs, without rerunning inference."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.suite import image_quality, quality_status, summarize


def rescore(source: Path, output: Path, corrections: Path) -> dict:
    if output.resolve() == source.resolve() or output.exists():
        raise ValueError("Use a new output directory; original evidence must remain intact")
    protocol = json.loads(corrections.read_text())
    summary = json.loads((source / "summary.json").read_text())
    records = [json.loads(s) for s in (source / "individual.jsonl").read_text().splitlines()]
    lookup = {r["id"]: r for r in records}
    for edit in protocol["images"]:
        row = lookup[edit["id"]]
        if row.get("key_phrases") != edit["expected_key_phrases"]:
            raise ValueError(f"Reference precondition failed: {edit['id']}")
        if row["stages"]["ocr"]["status"] != "ok":
            raise ValueError("Cannot rescore failed OCR as a reference-only correction")
        row["key_phrases"] = edit["key_phrases"]
        row["evaluation_correction"] = edit
        row["quality"] = image_quality(row, row["stages"]["ocr"].get("text", ""))
        row["quality_status"] = quality_status(row["quality"])
        row["responsible_stages"] = ["ocr"] if row["quality_status"] == "fail" else []
        # Execution/warnings remain unchanged; correcting a label does not erase them.
    queries = [json.loads(s) for s in (source / "queries.jsonl").read_text().splitlines()]
    excluded = {r["id"]: r for r in protocol["exclude_queries"]}
    if not set(excluded) <= {r["id"] for r in queries}:
        raise ValueError("Excluded query not found")
    retained = [r for r in queries if r["id"] not in excluded]
    grouped: dict[str, list] = defaultdict(list)
    for query in retained:
        for name, score in query["metrics"].items():
            for group in (
                "all",
                query["category"],
                "split:" + query["split"],
                query["category"] + ":" + query["split"],
                "language:" + query["language"],
                "kind:" + query["kind"],
            ):
                grouped[name + "/" + group].append(score)
    summary["retrieval"] = {
        key: {"queries": len(items), **{m: sum(r[m] for r in items) / len(items) for m in items[0]}}
        for key, items in sorted(grouped.items())
    }
    summary["aggregate"] = summarize(records)
    summary["config"]["evaluation_protocol_sha256"] = hashlib.sha256(
        corrections.read_bytes()
    ).hexdigest()
    summary["config"]["queries"] = [
        q for q in summary["config"]["queries"] if q["id"] not in excluded
    ]
    summary["evaluation_origin"] = str(source)
    summary["evaluation_corrections"] = protocol
    output.mkdir(parents=True)
    for name, rows in (("individual", records), ("queries", retained)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    (output / "excluded-queries.json").write_text(
        json.dumps(
            [{**q, "exclusion": excluded[q["id"]]} for q in queries if q["id"] in excluded],
            indent=2,
        )
        + "\n"
    )
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    args = parser.parse_args()
    rescore(args.source, args.output, args.corrections)


if __name__ == "__main__":
    main()
