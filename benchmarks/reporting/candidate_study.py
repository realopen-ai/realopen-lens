"""Publish compact, auditable evidence from the four-step candidate retrieval study."""

import argparse
from collections import Counter
import json
from pathlib import Path
import psutil

from benchmarks.datasets.photos import save_json
from realopen_lens.models.manager import digest
from realopen_lens.storage.catalog import Catalog
from realopen_lens.storage.artifacts import load_manifest


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path: Path) -> list[dict]:
    return list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def build(root: Path, output: Path) -> dict:
    depths = read(root / "depths/comparison.json")
    precision = read(root / "precision-retrieval-v3/report.json")
    expansion = read(root / "expansion-v2/report.json")
    lexical = read(root / "arabic-lexical/report.json")
    baseline = read(root / "depths/depth-50/config.json")
    corpus = read(root / "depths/depth-50/corpus.json")
    judgments = read(root / "depths/depth-50/judgments.json")
    probes = read(root / "precision-inputs/inputs.json")
    source = Path(__file__).resolve().parents[2]
    catalog = Catalog(Path(baseline["index_dir"]), readonly=True)
    try:
        _, generation = catalog.snapshot(baseline["collection"])
        manifest = load_manifest(catalog, generation)
    finally:
        catalog.close()
    if generation["id"] != baseline["snapshot"]["generation"]:
        raise ValueError("Study generation changed before publication")
    parity = {
        name: {
            stage: read(root / "parity" / name / stage / "report.json")
            for stage in ("vision", "siglip2", "e5-query", "e5-passage")
        }
        for name in ("int8-vs-fp32", "fp32-vs-reference")
    }
    parity["fp32-vs-reference-lowercase"] = {
        "siglip2": read(root / "parity/fp32-vs-reference-lowercase/siglip2/report.json")
    }
    ocr: dict = {}
    for name in ("baseline", "recovery", "band-recovery"):
        records = lines(root / "ocr" / name / "images.jsonl")
        ocr[name] = {
            "summary": read(root / "ocr" / name / "summary.json"),
            "observed_native_environments": [
                json.loads(value)
                for value in sorted(
                    {
                        json.dumps(
                            {key: r["actual"][key] for key in ("version", "language_pack_sha256")},
                            sort_keys=True,
                        )
                        for r in records
                        if r["status"] == "ok"
                    }
                )
            ],
            "individual": [
                {
                    "image_id": r["id"],
                    "category": r["input"]["category"],
                    "sha256": r["input"]["sha256"],
                    "status": r["status"],
                    "error_type": r.get("error_type"),
                    "seconds": r["seconds"],
                    "peak_tree_rss_mib": r["peak_tree_rss_mib"],
                    "quality_status": r["quality_status"],
                    "quality": r.get("quality"),
                    "characters": len(r.get("actual", {}).get("text", "")),
                    "timeout_recovery": r.get("actual", {}).get("timeout_recovery"),
                    "warnings": r.get("actual", {}).get("warnings", []),
                }
                for r in records
            ],
        }
    old = {r["id"]: r for r in lines(root / "ocr/baseline/images.jsonl")}
    final = {r["id"]: r for r in lines(root / "ocr/band-recovery/images.jsonl")}
    ocr["unchanged_successful_transcripts"] = [
        key
        for key, row in old.items()
        if row["status"] == "ok"
        and final[key]["status"] == "ok"
        and row["actual"]["text"] == final[key]["actual"]["text"]
    ]
    ocr["production_verification"] = read(root / "index-smoke/verification.json")
    for run in depths["runs"]:
        run["directory"] = Path(run["directory"]).name
    counts = Counter(row["image_id"] for row in probes["passages"])
    report = {
        "version": 1,
        "date": read(root / "protocol.json")["frozen_at"][:10],
        "scope": "Diagnostic only: previously inspected agent-reviewed queries; "
        "no new holdout or default promotion",
        "dataset": {
            "images": len(corpus["images"]),
            "queries": len(judgments["queries"]),
            "categories": dict(Counter(row["category"] for row in corpus["images"])),
            "corpus_sha256": baseline["corpus_sha256"],
            "judgments_sha256": baseline["judgments_sha256"],
        },
        "environment": {
            "host": baseline["host"],
            "packages": baseline["packages"],
            "threads": baseline["threads"],
            "host_ram_gib": psutil.virtual_memory().total / 2**30,
            "generation": baseline["snapshot"]["generation"],
            "model_registry": baseline["model_registry"],
            "cached_pipeline": baseline["snapshot"]["pipeline_spec"],
        },
        "depths": {"runs": depths["runs"], "individual": depths["paired_queries"]},
        "default_compatibility": read(root / "default-compatibility.json"),
        "probe_resume_verification": read(root / "probe-resume-verification.json"),
        "training_budget_verification": read(root / "training-budget-verification.json"),
        "installed_wheel_verification": read(root / "installed-wheel-verification.json"),
        "parity": parity,
        "reference_registry": read(source / "benchmarks/sources/encoder-reference.lock.json"),
        "reference_requirements_sha256": digest(
            source / "benchmarks/sources/encoder-reference.requirements.txt"
        ),
        "precision": {
            "aggregate": precision["aggregate"],
            "individual": [
                {
                    key: row[key]
                    for key in (
                        "query_id",
                        "policy",
                        "language",
                        "kind",
                        "pool_recall",
                        "positive_ranks",
                    )
                }
                | {"hybrid_metrics": row["metrics"]["hybrid"]}
                for row in precision["queries"]
            ],
            "probe_reports": precision["probe_reports"],
            "harness_sha256": precision["harness_sha256"],
            "engine_sha256": precision["engine_sha256"],
            "limitations": precision["limitations"],
        },
        "expansion": {
            "aggregate": expansion["aggregate"],
            "individual": [
                {key: value for key, value in row.items() if key != "rankings"}
                for row in expansion["queries"]
            ],
            "limitations": expansion["limitations"],
        },
        "arabic_lexical": {
            "aggregate": lexical["aggregate"],
            "individual": [
                {key: value for key, value in row.items() if key != "ranking"}
                for row in lexical["queries"]
            ],
            "limitations": lexical["limitations"],
        },
        "ocr": ocr,
        "remaining_pool_diagnostics": read(root / "diagnostics/remaining-pool-misses.json"),
        "passages": {
            "count": len(probes["passages"]),
            "maximum_per_image": max(counts.values()),
            "images_with_passages": len(counts),
            "configured_passage_cap": 16,
            "images_at_passage_cap": sum(value >= 16 for value in counts.values()),
            "maximum_observed_characters": max(len(row["text"]) for row in manifest["ocr"]),
            "empty_ocr_rows": sum(not row["text"] for row in manifest["ocr"]),
        },
        "evidence": {
            str(path.relative_to(root)): digest(path)
            for path in (
                root / "depths/comparison.json",
                root / "precision-retrieval-v3/report.json",
                root / "expansion-v2/report.json",
                root / "arabic-lexical/report.json",
                root / "ocr/band-recovery/images.jsonl",
                root / "index-smoke/verification.json",
                root / "probe-resume-verification.json",
                root / "training-budget-verification.json",
                root / "installed-wheel-verification.json",
            )
        },
        "limitations": [
            "Relevance judgments are incomplete and agent-reviewed, "
            "without independent native-speaker review.",
            "Depths are real production requests; "
            "precision and expansion are cached-evidence diagnostics.",
            "Earlier four/five-policy precision trials "
            "and the failed full-page OCR retry remain retained.",
            "No automatic translator, production FP32 policy, fresh holdout, "
            "native Windows/Linux inference, "
            "or 8 GB hardware validation is claimed.",
            "All large-scale retrieval comparisons reuse unchanged cached OCR; new OCR recovery is "
            "validated separately on 21 controls and a two-image production collection.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.run_dir, args.output)
    print(
        f"Published {report['dataset']['images']} images and {report['dataset']['queries']} queries"
    )


if __name__ == "__main__":
    main()
