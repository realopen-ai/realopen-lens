"""Diagnostic Arabic character folding; no stemming or production tokenizer change."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.metrics.collection import evaluate
from benchmarks.metrics.statistics import distribution
from realopen_lens.retrieval.ranking import lexical_index, rank_lexical
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog

FOLD = str.maketrans(
    {"ـ": "", "ى": "ي", "ی": "ي", "ک": "ك"}
    | {
        digit: str(i)
        for alphabet in ("٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹")
        for i, digit in enumerate(alphabet)
    }
)


def fold(text: str) -> str:
    """An additional experimental search view; retain original text for exact identifiers."""
    return text.translate(FOLD)


def run(index_dir: Path, collection: str, baseline_dir: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Preserve previous Arabic lexical probes; choose a new output directory")
    _, bundle = load(baseline_dir / "corpus.json", baseline_dir / "judgments.json")
    baseline = {
        r["query_id"]: r
        for r in map(
            json.loads, (baseline_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    catalog = Catalog(index_dir, readonly=True)
    try:
        _, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
    finally:
        catalog.close()
    if generation["id"] != config["snapshot"]["generation"]:
        raise ValueError("Arabic probe differs from baseline collection generation")
    started = perf_counter()
    records = [
        {
            **row,
            "text": fold(row["text"]),
            "alternate_texts": [fold(text) for text in row.get("alternate_texts", [])],
        }
        for row in manifest["ocr"]
    ]
    database = lexical_index(records)
    build_seconds = perf_counter() - started
    rows = []
    try:
        for query in bundle["queries"]:
            original = baseline[query["id"]]
            if original["status"] != "ok":
                raise ValueError("Lexical probe requires successful baseline requests")
            start = perf_counter()
            ranking = rank_lexical(database, fold(query["text"]), 50)
            labels = {row["image_sha256"]: row["grade"] for row in query["judgments"]}
            positives = {key for key, grade in labels.items() if grade > 0}
            original_pool = {r["image_id"] for r in original["trace"]["candidates"]}
            # Additional channel: original keyword candidates are not silently discarded.
            expanded = original_pool | set(ranking)
            rows.append(
                {
                    "query_id": query["id"],
                    "language": query["language"],
                    "ranking": ranking,
                    "metrics": evaluate(ranking, labels),
                    "original_metrics": evaluate(original["trace"]["rankings"]["lexical"], labels),
                    "new_pool_positives": sorted((expanded - original_pool) & positives),
                    "pool_recall": len(expanded & positives) / len(positives),
                    "original_pool_recall": len(original_pool & positives) / len(positives),
                    "seconds": perf_counter() - start,
                }
            )
    finally:
        database.close()
    report = {
        "version": 1,
        "baseline_identity": config,
        "index_build_seconds": build_seconds,
        "queries": rows,
        "aggregate": {
            language: {
                "queries": len(values),
                "new_pool_positives": sum(len(r["new_pool_positives"]) for r in values),
                "original_pool_recall": sum(r["original_pool_recall"] for r in values)
                / len(values),
                "pool_recall": sum(r["pool_recall"] for r in values) / len(values),
                "original_lexical_recall_at_50": sum(
                    r["original_metrics"]["candidate_recall_at_50"] for r in values
                )
                / len(values),
                "folded_lexical_recall_at_50": sum(
                    r["metrics"]["candidate_recall_at_50"] for r in values
                )
                / len(values),
                "lexical_probe_seconds": distribution([r["seconds"] for r in values]),
            }
            for language in ("en", "fr", "ar")
            if (values := [r for r in rows if r["language"] == language])
        },
        "limitations": [
            "Character folding is lossy; no production normalization change is made.",
            "These inspected queries are diagnostic and have incomplete judgments.",
            "Measures keyword and candidate coverage, not expanded hybrid ranking quality.",
            "Adds a separately normalized keyword channel with up to 50 candidates.",
        ],
    }
    output.mkdir(parents=True)
    save_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index-dir", "baseline-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--collection", required=True)
    report = run(**vars(parser.parse_args()))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
