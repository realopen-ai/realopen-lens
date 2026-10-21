"""Audit and evaluate explicitly reviewed feedback on an isolated index snapshot."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from benchmarks.experiments.virtual_collections import copy_snapshot
from realopen_lens.learning.dataset import canonical, prepare
from realopen_lens.learning.review import audit
from realopen_lens.learning.workflow import train
from realopen_lens.models.manager import digest
from realopen_lens.organization.duplicates import snapshot
from realopen_lens.storage.artifacts import atomic_bytes, load_manifest
from realopen_lens.storage.catalog import Catalog


def run(
    source: Path,
    output: Path,
    collection: str,
    review: Path | None,
    duplicate_index: Path | None = None,
    models: Path = Path("models"),
    on_progress=None,
) -> dict:
    """Never edit source feedback, rewrite source hashes or activate a model."""
    if output.exists():
        raise ValueError("Use a new experiment output directory")
    if not (source / "feedback.sqlite3").is_file():
        raise ValueError("Source index has no saved feedback")
    root = output / "index"
    if on_progress:
        on_progress({"phase": "snapshotting"})
    metadata = copy_snapshot(source, root, collection)
    with sqlite3.connect(
        (source / "feedback.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    ) as original:
        with sqlite3.connect(root / "feedback.sqlite3") as destination:
            original.backup(destination)
    catalog = Catalog(root, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
    finally:
        catalog.close()
    duplicates = duplicate_index or source
    grouped = snapshot(duplicates, manifest)
    if duplicate_index is not None and grouped is None:
        raise ValueError("Duplicate review index does not identify this source snapshot")
    if grouped is not None:
        # Copy only the published, matching snapshot and its source-scoped decisions.
        atomic_bytes(root / "duplicates" / (grouped["key"] + ".json"), canonical(grouped))
        path = duplicates / "duplicates/reviews" / (found["id"] + ".json")
        if path.exists():
            raw = path.read_bytes()
            atomic_bytes(root / "duplicates/reviews" / path.name, raw)
            metadata["duplicate_review_sha256"] = digest(root / "duplicates/reviews" / path.name)
        metadata["duplicate_snapshot"] = grouped["key"]
    metadata["feedback_snapshot_sha256"] = digest(root / "feedback.sqlite3")
    atomic_bytes(output / "snapshot.json", canonical(metadata))
    if on_progress:
        on_progress({"phase": "auditing"})
    report = audit(root, collection)
    atomic_bytes(output / "audit.json", canonical(report))
    if review is None:
        return {"status": "audited", "runs": len(report["runs"]), "cohorts": len(report["cohorts"])}
    frozen = prepare(root, review, output / "dataset", on_progress)
    result = train(output / "dataset/dataset.json", root, models, on_progress)
    if (root / "rerankers/state.json").exists():
        raise ValueError("Unexpected model activation state in isolated experiment")
    summary = {
        "prepared": frozen,
        "training": result,
        "interpretation": "diagnostic known judgments; no automatic model activation",
    }
    atomic_bytes(output / "result.json", canonical(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection", default="personal")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--duplicate-index", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()

    def progress(value):
        print(json.dumps(value, ensure_ascii=False), file=sys.stderr, flush=True)

    existed = args.output.exists()
    try:
        result = run(
            args.source_index,
            args.output,
            args.collection,
            args.review,
            args.duplicate_index,
            args.models,
            progress,
        )
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        if not existed and args.output.is_dir():
            atomic_bytes(
                args.output / "error.json", canonical({"error": f"{type(error).__name__}: {error}"})
            )
        raise


if __name__ == "__main__":
    main()
