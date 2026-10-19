"""Compare a small fixed-fusion grid using development judgments only."""

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.suite import connect, load_vectors, source_hash
from benchmarks.retrieval_metrics import metrics
from realopen_lens.retrieval import hybrid_rank, lexical_index, lexical_frequencies, rank_lexical


def run(root: Path, output: Path) -> list[dict]:
    config = json.loads((root / "config.json").read_text())
    db = connect(root / "results.sqlite")
    ids, visual = load_vectors(db, "vision")
    pids, semantic = load_vectors(db, "semantic")
    qids, qv = load_vectors(db, "visual_queries")
    sids, sv = load_vectors(db, "semantic_queries")
    qi = {key: i for i, key in enumerate(qids)}
    si = {key: i for i, key in enumerate(sids)}
    records = [
        {"image_id": key, **json.loads(payload)}
        for key, payload in db.execute("SELECT image_id,payload FROM stages WHERE stage='ocr'")
    ]
    for row in records:
        row.setdefault("text", "")
    index = lexical_index(records)
    queries = [q for q in config["queries"] if q["split"] == "dev"]
    candidates = {q["id"]: rank_lexical(index, q["text"]) for q in queries}
    statistics = {q["id"]: lexical_frequencies(index, q["text"]) for q in queries}
    rows: list[dict[str, Any]] = []
    for floor, margin in product((0.81, 0.83, 0.85), (0.02, 0.04, 0.08)):
        grouped: dict[str, list] = defaultdict(list)
        for query in queries:
            ranking = hybrid_rank(
                visual,
                qv[qi[query["id"]]],
                ids,
                semantic,
                sv[si[query["id"]]],
                pids,
                query["text"],
                records,
                semantic_floor=floor,
                semantic_margin=margin,
                lexical_candidates=candidates[query["id"]],
                lexical_statistics=statistics[query["id"]],
            )
            score = metrics(ranking, query["relevant"])
            grouped["all"].append(score)
            grouped[query["category"]].append(score)
            if query["kind"] == "natural":
                grouped["natural/" + query["category"]].append(score)
        result = {
            key: {
                "queries": len(items),
                **{metric: float(np.mean([r[metric] for r in items])) for metric in items[0]},
            }
            for key, items in grouped.items()
        }
        natural = [value for key, value in result.items() if key.startswith("natural/")]
        rows.append(
            {
                "semantic_floor": floor,
                "semantic_margin": margin,
                "selection_macro_hit5": float(np.mean([r["hit_rate_at_5"] for r in natural])),
                "selection_macro_mrr10": float(np.mean([r["mrr_at_10"] for r in natural])),
                "metrics": result,
            }
        )
    rows.sort(
        key=lambda r: (
            -r["selection_macro_hit5"],
            -r["selection_macro_mrr10"],
            -r["semantic_floor"],
            -r["semantic_margin"],
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Calibration output exists; choose a new file")
    output.write_text(
        json.dumps(
            {
                "source_run": str(root),
                "engine_sha256": source_hash(Path("src")),
                "dataset_sha256": config["manifest_sha256"],
                "split": "dev",
                "selection": (
                    "Equal-weight natural-query category Hit@5, then MRR@10; "
                    "conservative gates break ties"
                ),
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    import shutil

    shutil.copy2(Path("src/realopen_lens/retrieval.py"), output.with_suffix(".retrieval.py"))
    index.close()
    db.close()
    print(
        json.dumps(
            [
                {
                    key: r[key]
                    for key in (
                        "semantic_floor",
                        "semantic_margin",
                        "selection_macro_hit5",
                        "selection_macro_mrr10",
                    )
                }
                for r in rows
            ],
            indent=2,
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.output)


if __name__ == "__main__":
    main()
