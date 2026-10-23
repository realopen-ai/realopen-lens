"""Replay verified production query embeddings to diagnose mixed precision fusion.

Replay timings exclude query inference and are never reported as runtime latency.
"""

import argparse
import json
from pathlib import Path
from importlib.metadata import version
from unittest.mock import patch

import numpy as np

from benchmarks.experiments.precision_retrieval import vectors
from benchmarks.metrics.collection import evaluate
from realopen_lens.models.manager import digest
from realopen_lens.retrieval import library, ranking


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = json.loads(args.judgments.read_text(encoding="utf-8"))
    queries = bundle["queries"]
    baseline_config = json.loads((args.baseline / "config.json").read_text(encoding="utf-8"))
    if baseline_config["judgments_sha256"] != digest(args.judgments):
        raise ValueError("Baseline judgments changed")
    if args.query_id:
        wanted = set(args.query_id)
        queries = [q for q in queries if q["id"] in wanted]
        if {q["id"] for q in queries} != wanted:
            raise ValueError("Unknown replay query ID")
    inventory = json.loads(args.inputs.read_text(encoding="utf-8"))["queries"]
    by_text = {row["text"]: i for i, row in enumerate(inventory)}
    cached = {}
    reports = {}
    for precision, kind, stage in [
        ("int8", "siglip2", "siglip2"),
        ("fp32", "siglip2", "siglip2"),
        ("int8", "e5", "e5-query"),
    ]:
        array, report = vectors(args.probes / precision / stage, inventory)
        if report["identity"]["inputs_sha256"] != digest(args.inputs):
            raise ValueError("Query probe inventory checksum changed")
        if report["identity"]["inference_sha256"] != digest(
            Path("src/realopen_lens/encoders/onnx.py")
        ):
            raise ValueError("Query preprocessing changed")
        for name in ("numpy", "onnxruntime", "tokenizers"):
            if report["identity"]["packages"][name] != version(name):
                raise ValueError("Query probe dependency versions changed")
        cached[precision, kind] = array
        reports[precision + "/" + kind] = report
    config = {
        "judgments_sha256": digest(args.judgments),
        "inputs_sha256": digest(args.inputs),
        "probes": reports,
        "index": str(args.index.resolve()),
        "mode": "cached-query replay; no inference latency claims",
        "query_ids": [q["id"] for q in queries],
        "sources": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("src/realopen_lens/retrieval/library.py"),
                Path("src/realopen_lens/retrieval/ranking.py"),
            )
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    current = {}

    class Encoder:
        def verify(self, models, threads, precision):
            current["precision"] = precision

        def encode(self, text, kind, models, threads, precision):
            return cached[(precision if kind == "siglip2" else "int8", kind)][by_text[text]]

        def release(self):
            pass

        def close(self):
            pass

    raw = ranking.hybrid_scores
    records = []

    def fusion(visual, semantic, text, ocr, **kwargs):
        baseline_trace: dict = {}
        baseline = raw(visual, semantic, text, ocr, **{**kwargs, "trace": baseline_trace})
        precision_groups = {
            p: [key for key in visual if (int(key, 16) % 2 == 1) == (p == "fp32")]
            for p in ("int8", "fp32")
        }
        # The evaluation workspace is a frozen hash-modulo partition. No labels
        # enter normalization, candidate generation or score merging.
        variants = {"raw": baseline}
        reference = np.asarray([visual[k] for k in precision_groups["fp32"]])
        for method in ("moments", "robust"):
            transformed = visual.copy()

            def stats(values):
                if method == "moments":
                    return float(np.mean(values)), float(np.std(values))
                center = float(np.median(values))
                return center, float(np.median(np.abs(values - center)))

            ref_center, ref_scale = stats(reference)
            values = np.asarray([visual[k] for k in precision_groups["int8"]])
            center, scale = stats(values)
            if scale > 1e-8 and ref_scale > 1e-8:
                for key in precision_groups["int8"]:
                    transformed[key] = (visual[key] - center) / scale * ref_scale + ref_center
            variants[method] = raw(transformed, semantic, text, ocr, **{**kwargs, "trace": None})
        local = {}
        local_scores = {}
        local_evidence = {}
        for precision, keys in precision_groups.items():
            local_trace: dict = {}
            local[precision] = raw(
                {k: visual[k] for k in keys},
                semantic,
                text,
                ocr,
                **{**kwargs, "trace": local_trace},
            )
            for c in local_trace["candidates"]:
                local_scores[c["image_id"]] = c["hybrid_score"]
                local_evidence[c["image_id"]] = max(c["lexical_evidence"], c["semantic_strength"])
        ranks = {k: i for ids in local.values() for i, k in enumerate(ids, 1)}
        variants["group-rank"] = sorted(ranks, key=lambda k: (ranks[k], -local_scores[k], k))
        variants["group-rank-evidence"] = sorted(
            ranks, key=lambda k: (-int(local_evidence[k] >= 0.8), ranks[k], -local_scores[k], k)
        )
        current["variants"] = variants
        current["trace"] = baseline_trace
        current["visual"] = visual
        current["semantic"] = semantic
        return baseline

    recorded = {
        r["query_id"]: r
        for r in map(
            json.loads, (args.baseline / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    with (
        patch.object(library, "ResidentQueryEncoder", Encoder),
        patch.object(ranking, "hybrid_scores", fusion),
    ):
        session = library.LibrarySession(index_dir=args.index)
        try:
            with (args.output / "queries.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                for query in queries:
                    if query["text"] not in by_text:
                        raise ValueError("Missing exact query text")
                    trace: dict = {}
                    session.search(query["text"], limit=50, use_feedback=False, trace=trace)
                    labels = {j["image_sha256"]: j["grade"] for j in query["judgments"]}
                    expected = recorded[query["id"]]
                    actual = current["variants"]["raw"][:50]
                    if actual != [r["image_id"] for r in expected["results"]]:
                        raise ValueError(
                            "Replay does not reproduce recorded result ranking: " + query["id"]
                        )
                    old = {r["image_id"]: r for r in expected["trace"]["candidates"]}
                    for c in current["trace"]["candidates"]:
                        for key in ("visual_cosine", "semantic_cosine", "hybrid_score"):
                            if c[key] is not None and old[c["image_id"]][key] is not None:
                                if abs(c[key] - old[c["image_id"]][key]) > 1e-6:
                                    raise ValueError("Replay scores changed")
                    row = {
                        "query_id": query["id"],
                        "split": query["split"],
                        "language": query["language"],
                        "metrics": {
                            name: evaluate(ids[:50], labels)
                            for name, ids in current["variants"].items()
                        },
                        "rankings": {name: ids[:50] for name, ids in current["variants"].items()},
                        "baseline_parity": True,
                    }
                    records.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    np.savez_compressed(
                        args.output / (query["id"] + ".npz"),
                        ids=np.asarray(list(current["visual"])),
                        visual=np.asarray(list(current["visual"].values())),
                        semantic_ids=np.asarray(list(current["semantic"])),
                        semantic=np.asarray(list(current["semantic"].values())),
                    )
        finally:
            session.close()
    result = {}
    for split in ("all", "train", "validation", "test"):
        rows = [r for r in records if split == "all" or r["split"] == split]
        if not rows:
            continue
        result[split] = {
            name: {
                key: float(np.mean([r["metrics"][name][key] for r in rows]))
                for key in ("candidate_recall_at_50", "mrr_at_10", "ndcg_at_10")
            }
            for name in records[0]["metrics"]
        }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", action="append", help="Replay only these frozen IDs")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument(
        "--judgments", type=Path, default=Path("benchmarks/sources/retrieval-v2/judgments.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
