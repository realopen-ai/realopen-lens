"""Preserve per-query lexical and multilingual E5 diagnostic rankings."""

import argparse
import json
import importlib.util
from typing import Any
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.datasets.multilingual import SOURCE, digest
from benchmarks.metrics.retrieval import metrics
from realopen_lens.retrieval import ranking


def run(
    dataset: Path,
    output: Path,
    ocr_run: Path | None = None,
    models: Path | None = None,
    ranking_module: Any = None,
):
    engine: Any = ranking_module or ranking
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["images"]
    records = [{"image_id": r["id"], "text": r["text"]} for r in rows]
    cases = json.loads((SOURCE / "lexical.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "ranking_sha256": digest(Path(engine.__file__)),
        "manifest_sha256": digest(dataset / "manifest.json"),
        "cases": [],
        "queries": [],
    }
    for case in cases:
        index = engine.lexical_index(
            [
                {"image_id": "a-distractor", "text": case["distractor"]},
                {"image_id": "z-target", "text": case["text"]},
            ]
        )
        try:
            ranked = engine.rank_lexical(index, case["query"])
            result["cases"].append(
                {
                    **case,
                    "ranking": ranked,
                    "pass": ranked == ["z-target"],
                    "target_at_1": ranked[:1] == ["z-target"],
                }
            )
        finally:
            index.close()
    corpora = {"reference": records}
    if ocr_run:
        observed = [
            json.loads(line)
            for line in (ocr_run / "results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for profile in sorted({r["profile"] for r in observed}):
            corpora[profile] = [
                {"image_id": r["id"], "text": r["actual"].get("text", "")}
                for r in observed
                if r["profile"] == profile
            ]
    encoder = None
    if models:
        from realopen_lens.encoders.onnx import TextEncoder
        from realopen_lens.models.manager import artifact_path, registry

        spec = registry()["models"]["e5"]
        actual = {
            key: digest(artifact_path(models, "e5", key)) for key in ("text_int8", "tokenizer")
        }
        if any(actual[key] != spec["files"][key]["sha256"] for key in actual):
            raise ValueError("E5 model integrity mismatch")
        result["e5"] = {"registry": spec, "actual_sha256": actual}

        encoder = TextEncoder(
            artifact_path(models, "e5", "text_int8"),
            artifact_path(models, "e5", "tokenizer"),
            "e5",
            2,
        )
    queries = [row["query"] for row in rows]
    query_vectors = (
        np.concatenate([encoder.encode(queries[i : i + 4]) for i in range(0, len(queries), 4)])
        if encoder
        else None
    )
    for name, corpus in corpora.items():
        index = engine.lexical_index(corpus)
        vectors = None
        if encoder:
            texts = [r["text"] or "[no OCR text]" for r in corpus]
            vectors = np.concatenate(
                [encoder.encode(texts[i : i + 4], passage=True) for i in range(0, len(texts), 4)]
            )
        for i, row in enumerate(rows):
            start = perf_counter()
            lexical = engine.rank_lexical(index, row["query"], 50)
            record = {
                "id": row["id"],
                "language": row["language"],
                "corpus": name,
                "query": row["query"],
                "lexical_ranking": lexical,
                "lexical": metrics(lexical, [row["id"]]),
                "seconds": perf_counter() - start,
            }
            if vectors is not None and query_vectors is not None:
                semantic = engine.rank_vectors(
                    vectors, query_vectors[i], [r["image_id"] for r in corpus], len(corpus)
                )
                lookup = {r["id"]: r for r in rows}
                cross = [key for key in semantic if lookup[key]["language"] != row["language"]]
                relevant = [
                    r["id"]
                    for r in rows
                    if r["topic"] == row["topic"] and r["language"] != row["language"]
                ]
                record.update(semantic_cross_ranking=cross, semantic_cross=metrics(cross, relevant))
            result["queries"].append(record)
        index.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Lexical regression cases: {sum(r['pass'] for r in result['cases'])}/{len(cases)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ocr-run", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument(
        "--ranking-source",
        type=Path,
        help="Explicit reviewed historical source for paired comparison",
    )
    args = parser.parse_args()
    legacy = None
    if args.ranking_source:
        spec = importlib.util.spec_from_file_location(
            "multilingual_baseline_ranking", args.ranking_source
        )
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load baseline source")
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
    run(args.dataset, args.output, args.ocr_run, args.models, legacy)


if __name__ == "__main__":
    main()
