"""Small-benchmark retrieval evaluation and Markdown reporting."""

import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
from benchmarks.metrics.retrieval import metrics
from realopen_lens.retrieval.ranking import (
    hybrid_rank,
    lexical_index,
    lexical_frequencies,
    rank_lexical,
    rank_vectors,
)


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["output"])
    if not config["queries"]:
        return {
            "queries": [],
            "aggregates": {},
            "aggregates_by_category": {},
            "note": "No query judgments supplied",
        }
    vision = np.load(output / "vision.npy", mmap_mode="r", allow_pickle=False)
    visual_queries = np.load(output / "visual_queries.npy", allow_pickle=False)
    semantic_queries = np.load(output / "semantic.npy", allow_pickle=False)
    passages = json.loads((output / "passages.json").read_text())
    vectors = (
        np.load(output / "passages.npy", mmap_mode="r", allow_pickle=False) if passages else None
    )
    records = json.loads((output / "ocr.json").read_text())["records"]
    database = lexical_index(records)
    results = []
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    categories: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    try:
        for i, query in enumerate(config["queries"]):
            first_start = perf_counter()
            visual = rank_vectors(vision, visual_queries[i], config["images"])
            lexical = rank_lexical(database, query["text"])
            semantic = (
                rank_vectors(vectors, semantic_queries[i], [item["image_id"] for item in passages])
                if vectors is not None
                else []
            )
            rankings = {
                "visual": visual,
                "lexical": lexical,
                "semantic_ocr": semantic,
                "hybrid": hybrid_rank(
                    vision,
                    visual_queries[i],
                    config["images"],
                    vectors if vectors is not None else np.empty((0, 0)),
                    semantic_queries[i],
                    [item["image_id"] for item in passages],
                    query["text"],
                    records,
                    lexical_candidates=lexical,
                    lexical_statistics=lexical_frequencies(database, query["text"]),
                ),
            }
            first_seconds = perf_counter() - first_start
            repeat_seconds = []
            for _ in range(config["repeats"]):
                start = perf_counter()
                rank_vectors(vision, visual_queries[i], config["images"])
                rank_lexical(database, query["text"])
                if vectors is not None:
                    rank_vectors(vectors, semantic_queries[i], [p["image_id"] for p in passages])
                hybrid_rank(
                    vision,
                    visual_queries[i],
                    config["images"],
                    vectors if vectors is not None else np.empty((0, 0)),
                    semantic_queries[i],
                    [p["image_id"] for p in passages],
                    query["text"],
                    records,
                    lexical_candidates=lexical,
                    lexical_statistics=lexical_frequencies(database, query["text"]),
                )
                repeat_seconds.append(perf_counter() - start)
            scores = {
                name: metrics(ranking, query["relevant"]) for name, ranking in rankings.items()
            }
            results.append(
                {
                    "id": query["id"],
                    "language": query["language"],
                    "category": query.get("category"),
                    "rankings": rankings,
                    "metrics": scores,
                    "retrieval_first_seconds": first_seconds,
                    "retrieval_repeat_seconds": repeat_seconds,
                }
            )
            for name, score in scores.items():
                if score:
                    grouped[name, "all"].append(score)
                    grouped[name, query["language"]].append(score)
                    if query.get("category"):
                        categories[name, query["category"]].append(score)
    finally:
        database.close()
    aggregates: dict[str, dict[str, Any]] = defaultdict(dict)
    for (name, language), items in grouped.items():
        aggregates[name][language] = {
            "query_count": len(items),
            **{metric: sum(item[metric] for item in items) / len(items) for metric in items[0]},
        }
    return {
        "queries": results,
        "aggregates": dict(aggregates),
        "aggregates_by_category": aggregate_categories(categories),
        "judgment_policy": "Metrics use supplied positives; other results remain unjudged.",
    }


def aggregate_categories(groups: dict[tuple[str, str], list[dict[str, float]]]) -> dict[str, Any]:
    result: dict[str, dict[str, Any]] = defaultdict(dict)
    for (channel, category), items in groups.items():
        result[channel][category] = {
            "query_count": len(items),
            **{key: sum(item[key] for item in items) / len(items) for key in items[0]},
        }
    return dict(result)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local inference benchmark",
        "",
        f"Run: {report['created_at']}",
        "",
        f"Corpus: {report['corpus_kind']}; {report['image_count']} images; "
        f"{report['query_count']} queries.",
        "",
        f"Host: {report['host']['system']} {report['host']['machine']}; "
        f"{report['host']['ram_gib']:.1f} GiB RAM.",
        "",
        f"Precision: {report['precision']}; CPU; batch size {report['batch_size']}; "
        f"{report['threads']} threads.",
        "",
        f"Sampled peak process-tree RSS: {report['memory']['peak_mib']:.1f} MiB "
        "(20 ms samples; shared pages may be counted more than once).",
        "",
        "This run does not establish support for other hardware or languages. "
        "Generated fixtures only test pipeline execution.",
        "",
        "## Stages",
        "",
        "| Stage | Wall seconds | Peak tree RSS (MiB) |",
        "| --- | ---: | ---: |",
    ]
    for name, stage in report["stages"].items():
        lines.append(f"| {name} | {stage['wall_seconds']:.3f} | {stage['peak_mib']:.1f} |")
    lines += [
        "",
        "Model loading, first query execution, repeated query execution, OCR timings, "
        "and retrieval timings are stored separately in report.json. Query encoding "
        "timings exclude model load; they are not end-to-end search latency.",
        "",
        "## Retrieval",
        "",
        "Metrics are relative to supplied relevance judgments; incomplete labels "
        "do not establish true corpus recall.",
        "",
        "| Channel | Language | Queries | Recall@5 | Hit@5 | MRR@10 | nDCG@10 | Recall@50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for channel, languages in report["retrieval"]["aggregates"].items():
        for language, score in languages.items():
            values = " | ".join(
                f"{score[k]:.3f}"
                for k in [
                    "recall_at_5",
                    "hit_rate_at_5",
                    "mrr_at_10",
                    "ndcg_at_10",
                    "candidate_recall_at_50",
                ]
            )
            lines.append(f"| {channel} | {language} | {score['query_count']} | {values} |")
    lines += [
        "",
        "## OCR accuracy",
        "",
        "Character and word error rates use NFKC, case folding, and collapsed whitespace. "
        "Only supplied transcriptions are scored; empty reference denominators are undefined.",
        "",
        "| Language | Images | CER | WER |",
        "| --- | ---: | ---: | ---: |",
    ]
    for language, score in report["ocr_accuracy"].items():
        cer = f"{score['cer']:.3f}" if score["cer"] is not None else "undefined"
        wer = f"{score['wer']:.3f}" if score["wer"] is not None else "undefined"
        lines.append(f"| {language} | {score['images']} | {cer} | {wer} |")
    if not report["ocr_accuracy"]:
        lines += ["", "No reference OCR transcriptions were supplied."]
    if report["ocr_accuracy_by_category"]:
        lines += [
            "",
            "## OCR by image category",
            "",
            "Token recall counts matching word occurrences regardless of reading order. "
            "Key phrase recall uses normalized exact substrings; punctuation is retained. "
            "CER/WER remain sensitive to reading order, especially for diagrams.",
            "",
            "| Category | Images | CER | WER | Token recall | Key phrase recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for category, score in report["ocr_accuracy_by_category"].items():
            values = " | ".join(
                f"{score[key]:.3f}" if score[key] is not None else "undefined"
                for key in ("cer", "wer", "token_recall", "key_phrase_recall")
            )
            lines.append(f"| {category} | {score['images']} | {values} |")
    if report["retrieval"]["aggregates_by_category"]:
        lines += [
            "",
            "## Retrieval by image category",
            "",
            "| Channel | Category | Queries | Hit@5 | MRR@10 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for channel, groups in report["retrieval"]["aggregates_by_category"].items():
            for category, score in groups.items():
                lines.append(
                    f"| {channel} | {category} | {score['query_count']} | "
                    f"{score['hit_rate_at_5']:.3f} | {score['mrr_at_10']:.3f} |"
                )
    lines += [
        "",
        "## Offline policy",
        "",
        "Artifacts were loaded from verified local paths. Python network operations "
        "were blocked by audit hooks in model workers. This is not an OS firewall. "
        "OS-level blocked-network verification must be performed separately.",
        "",
        "Reports, OCR text, embeddings, query judgments, and worker logs are private "
        "local artifacts; review them before sharing.",
        "",
    ]
    return "\n".join(lines)
