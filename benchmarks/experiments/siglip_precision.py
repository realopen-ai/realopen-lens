"""Compare production INT8/FP32 collections on a deterministic reviewed corpus subset."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.experiments.collections import run as evaluate_collection
from benchmarks.runtime.resources import MemoryMonitor, write_resources
from realopen_lens.indexing import index_collection
from realopen_lens.runtime.offline import enforce


def freeze(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Subset identity changed; use a new output directory")
    else:
        save_json(path, value)


def prepare(args) -> tuple[Path, Path, Path]:
    corpus, bundle = load(args.corpus, args.judgments)
    groups = defaultdict(list)
    for query in bundle["queries"]:
        if query["language"] == "en":
            groups[query.get("kind", "mixed")].append(query)
    rng = random.Random(args.seed)
    families: set[str] = set()
    for kind in sorted(groups):
        choices = sorted(groups[kind], key=lambda q: q["id"])
        rng.shuffle(choices)
        families.update(q["family"] for q in choices[: args.families_per_kind])
    queries = [q for q in bundle["queries"] if q["family"] in families]
    # Preserve every reviewed label for selected families, including negatives.
    required = {j["image_sha256"] for q in queries for j in q["judgments"]}
    if not queries or len(required) > args.image_count or args.image_count > len(corpus["images"]):
        raise ValueError("Image count must fit the selected reviewed labels and source corpus")
    remaining = sorted(
        (r for r in corpus["images"] if r["sha256"] not in required), key=lambda r: r["id"]
    )
    rng.shuffle(remaining)
    selected = required | {r["sha256"] for r in remaining[: args.image_count - len(required)]}
    images = [r for r in corpus["images"] if r["sha256"] in selected]
    root = args.output / "images"
    subset = args.output / "subset"
    subset.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "seed": args.seed,
        "families_per_kind": args.families_per_kind,
        "source_corpus_sha256": digest(args.corpus),
        "source_judgments_sha256": digest(args.judgments),
        "scope": "Diagnostic subset of previously reviewed judgments; not a fresh holdout",
        "images": images,
    }
    corpus_path, judgments_path = subset / "corpus.json", subset / "judgments.json"
    freeze(corpus_path, manifest)
    freeze(judgments_path, {"version": 1, "corpus_sha256": digest(corpus_path), "queries": queries})
    source_root = args.source_root.resolve()
    for row in images:
        source, target = (source_root / row["path"]).resolve(), (root / row["path"]).resolve()
        if not source.is_relative_to(source_root) or not target.is_relative_to(root.resolve()):
            raise ValueError("Corpus image path escapes its root")
        if digest(source) != row["sha256"]:
            raise ValueError("Source image checksum changed")
        if target.exists():
            if digest(target) != row["sha256"]:
                raise ValueError("Prepared image changed; use a new output directory")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return root, corpus_path, judgments_path


def run(args) -> dict:
    root, corpus, judgments = prepare(args)
    reports = {}
    for precision in ("int8", "fp32"):
        output = args.output / precision
        output.mkdir(exist_ok=True)
        memory = MemoryMonitor()
        memory.phase = "production_index"
        with memory:
            result = index_collection(
                root,
                index_dir=output / "index",
                collection="precision-study",
                models=args.models,
                threads=args.threads,
                languages=args.languages,
                siglip_precision=precision,
                log=True,
            )
        write_resources(output, memory)
        attempt = (
            output / f"index-attempt-{len(list(output.glob('index-attempt-*.json'))) + 1:03}.json"
        )
        save_json(attempt, asdict(result))
        if not (output / "index-initial.json").exists():
            save_json(output / "index-initial.json", asdict(result))
        if result.failed_files or result.images != args.image_count:
            raise ValueError("Index did not publish every selected image; inspect index attempt")
        evaluate_collection(
            SimpleNamespace(
                corpus=corpus,
                judgments=judgments,
                index_dir=output / "index",
                collection="precision-study",
                models=args.models,
                threads=args.threads,
                output=output / "evaluation",
                retry_errors=args.retry_errors,
            )
        )
        reports[precision] = {
            "initial_index": json.loads(
                (output / "index-initial.json").read_text(encoding="utf-8")
            ),
            "index_resources": json.loads((output / "resources.json").read_text(encoding="utf-8")),
            "evaluation": json.loads(
                (output / "evaluation/summary.json").read_text(encoding="utf-8")
            ),
        }
        records = [
            json.loads(line)
            for line in (output / "evaluation/queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        errors = sum(r["memory_access_errors"] for r in records)
        index_errors = reports[precision]["index_resources"]["access_errors"]
        reports[precision]["memory_measurement"] = {
            "complete": errors == 0 and index_errors == 0,
            "query_access_errors": errors,
            "index_access_errors": index_errors,
            "limitation": (
                "RSS figures are incomplete when access errors are nonzero; "
                "do not interpret coordinator-only samples as model memory usage"
            ),
        }
    images = json.loads(corpus.read_text(encoding="utf-8"))["images"]
    report = {
        "scope": "Production separate-index diagnostic; no fresh holdout or platform certification",
        "images": len(images),
        "categories": dict(Counter(r["category"] for r in images)),
        "policies": reports,
    }
    save_json(args.output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--image-count", type=int, default=128)
    parser.add_argument("--families-per-kind", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--languages", default="eng+fra+ara")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.image_count < 1 or args.families_per_kind < 1:
        parser.error("Image count and families per kind must be positive")
    enforce()
    run(args)


if __name__ == "__main__":
    main()
