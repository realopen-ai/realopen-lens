"""Per-query retrieval rankings and grouped quality metrics."""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from benchmarks.metrics.retrieval import metrics
from benchmarks.storage.results import load_vectors


def retrieval_report(db: sqlite3.Connection, config: dict, output: Path) -> dict:
    from realopen_lens.retrieval.ranking import fuse, lexical_index, rank_lexical, rank_vectors

    ids, visual = load_vectors(db, "vision")
    pids, semantic = load_vectors(db, "semantic")
    qids, qv = load_vectors(db, "visual_queries")
    sids, qs = load_vectors(db, "semantic_queries")
    lookup = {q: i for i, q in enumerate(qids)}
    slookup = {q: i for i, q in enumerate(sids)}
    records = [
        {
            "image_id": key,
            "text": json.loads(payload).get("text", ""),
            "alternate_texts": json.loads(payload).get("alternate_texts", []),
        }
        for key, payload in db.execute("SELECT image_id,payload FROM stages WHERE stage='ocr'")
    ]
    index = lexical_index(records)
    results = []
    grouped: dict[str, list] = defaultdict(list)
    for query in config["queries"]:
        failed = query["id"] not in lookup or query["id"] not in slookup
        if failed:
            rankings: dict[str, list[str]] = {
                name: [] for name in ("visual", "lexical", "semantic_ocr", "hybrid")
            }
        else:
            v = rank_vectors(visual, qv[lookup[query["id"]]], ids, limit=len(ids))
            lex = rank_lexical(index, query["text"])
            sem = (
                rank_vectors(semantic, qs[slookup[query["id"]]], pids, limit=len(set(pids)))
                if len(pids)
                else []
            )
            if config["variant"] == "baseline":
                hybrid = fuse([v[:50], lex[:50], sem[:50]])
            else:
                from realopen_lens.retrieval.ranking import hybrid_rank, lexical_frequencies

                hybrid = hybrid_rank(
                    visual,
                    qv[lookup[query["id"]]],
                    ids,
                    semantic,
                    qs[slookup[query["id"]]],
                    pids,
                    query["text"],
                    records,
                    lexical_candidates=lex,
                    lexical_statistics=lexical_frequencies(index, query["text"]),
                )
            rankings = {"visual": v, "lexical": lex, "semantic_ocr": sem, "hybrid": hybrid}
        scores = {name: metrics(r, query["relevant"]) for name, r in rankings.items()}
        result = {
            **query,
            "status": "error" if failed else "ok",
            "error": "Query embedding absent" if failed else None,
            "rankings": {k: r[:50] for k, r in rankings.items()},
            "metrics": scores,
        }
        results.append(result)
        for name, score in scores.items():
            for group in (
                "all",
                query["category"],
                "split:" + query["split"],
                query["category"] + ":" + query["split"],
                "language:" + query["language"],
                "kind:" + query["kind"],
                *(["corpus:" + query["corpus"]] if "corpus" in query else []),
            ):
                grouped[name + "/" + group].append(score)
    index.close()
    (output / "queries.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    )
    return {
        key: {"queries": len(items), **{m: sum(r[m] for r in items) / len(items) for m in items[0]}}
        for key, items in sorted(grouped.items())
    }
