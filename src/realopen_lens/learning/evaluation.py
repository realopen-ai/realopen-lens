"""Known-judgment metrics with explicit coverage and per-query production evidence."""

from typing import Any
import math
from time import perf_counter
from pathlib import Path

import numpy as np
import psutil

from realopen_lens.learning.dataset import canonical
from realopen_lens.learning.logistic import load
from realopen_lens.retrieval.session import SearchSession

METRICS = ("candidate_recall_at_50", "mrr_at_10", "ndcg_at_10", "judged_fraction_at_10")


def quality(ranking: list[str], labels: dict[str, int]) -> dict:
    positives = {key for key, value in labels.items() if value == 1}
    if not positives or len(ranking) != len(set(ranking)):
        raise ValueError("Evaluation needs known positives and unique ranked images")
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(positives))))
    return {
        "candidate_recall_at_50": len(set(ranking[:50]) & positives) / len(positives),
        "mrr_at_10": next(
            (1 / i for i, key in enumerate(ranking[:10], 1) if key in positives), 0.0
        ),
        "ndcg_at_10": sum(
            1 / math.log2(i + 2) for i, key in enumerate(ranking[:10]) if key in positives
        )
        / ideal,
        "judged_fraction_at_10": sum(key in labels for key in ranking[:10])
        / max(1, min(10, len(ranking))),
    }


def aggregate(records: list[dict]) -> dict:
    good = [r for r in records if r["status"] == "ok"]
    result: dict[str, Any] = {
        "queries": len(records),
        "evaluated": len(good),
        "failed": len(records) - len(good),
    }
    if good:
        for name in ("hybrid", "logistic"):
            result[name] = {key: float(np.mean([r[name][key] for r in good])) for key in METRICS}
    return result


def replay(model, sessions: list[dict]) -> list[dict]:
    records = []
    for session in sessions:
        candidates = session["candidates"]
        hybrid = [
            r["image_id"]
            for r in sorted(candidates, key=lambda c: (-c["hybrid_score"], c["image_id"]))
        ]
        records.append(
            {
                "run_id": session["run_id"],
                "status": "ok",
                "hybrid": quality(hybrid, session["labels"]),
                "logistic": quality(model.rank(candidates), session["labels"]),
            }
        )
    return records


def production(
    dataset: dict, model: Path, index_dir: Path, models: Path, output: Path, on_progress=None
) -> dict:
    """Evaluate once after model selection; failures remain individual report rows."""
    records = []
    process = psutil.Process()
    load(model)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        evaluated = [session for session in dataset["sessions"] if session["split"] != "train"]
        for position, session in enumerate(evaluated):
            if on_progress:
                on_progress(
                    {
                        "phase": "evaluating",
                        "query": position + 1,
                        "queries": len(evaluated),
                    }
                )
            record = {
                "run_id": session["run_id"],
                "split": session["split"],
                "family": session["family"],
                "query": session["query"],
                "labels": session["labels"],
                "status": "failed",
            }
            sampled_rss = []
            memory_errors = []

            def observe(value):
                resident = 0
                try:
                    members = [process, *process.children(recursive=True)]
                except (psutil.Error, OSError) as error:
                    members = [process]
                    memory_errors.append(
                        f"Process-tree sampling unavailable: {type(error).__name__}"
                    )
                for member in members:
                    try:
                        resident += member.memory_info().rss
                    except (psutil.Error, OSError) as error:
                        memory_errors.append(f"RSS sampling unavailable: {type(error).__name__}")
                sampled_rss.append(resident)
                if on_progress:
                    on_progress(value)

            started = perf_counter()
            try:
                trace: dict = {}
                collection = session["collection"]
                if session.get("collection_id"):
                    from realopen_lens.storage.catalog import Catalog

                    catalog = Catalog(index_dir, readonly=True)
                    try:
                        collection = catalog.collection_name(session["collection_id"])
                    finally:
                        catalog.close()
                with SearchSession(
                    collection=collection, index_dir=index_dir, models=models
                ) as search:
                    latencies = {}
                    for variant in (
                        ("hybrid", "logistic") if position % 2 == 0 else ("logistic", "hybrid")
                    ):
                        variant_started = perf_counter()
                        observed: dict = {}
                        result = search.search(
                            session["query"]["text"],
                            use_feedback=False,
                            reranker=model if variant == "logistic" else None,
                            use_active_reranker=False,
                            trace=observed,
                            limit=50,
                            on_progress=observe,
                        )
                        latencies[variant] = perf_counter() - variant_started
                        if variant == "logistic":
                            trace = observed
                            results = result
                        else:
                            baseline = [r.image_id for r in result]
                            baseline_generation = observed["generation"]
                    record["latency_seconds"] = latencies
                    record["execution_order"] = (
                        "hybrid_first" if position % 2 == 0 else "logistic_first"
                    )
                if baseline_generation != trace["generation"]:
                    raise ValueError("Collection generation changed between paired requests")
                if trace["generation"] != session["generation"]:
                    raise ValueError("Collection generation changed since dataset freeze")
                if trace["siglip_precision"] != dataset["compatibility"].get(
                    "siglip_precision", "int8"
                ):
                    raise ValueError("Production evaluation requires matching encoder precision")
                actual = [r.image_id for r in results]
                record.update(
                    status="ok",
                    hybrid=quality(baseline, session["labels"]),
                    logistic=quality(actual, session["labels"]),
                    hybrid_ranking=baseline,
                    logistic_ranking=actual,
                    trace=trace,
                )
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            record.update(
                seconds=perf_counter() - started,
                rss_after_bytes=process.memory_info().rss,
                sampled_process_tree_rss_max_bytes=max(sampled_rss, default=0),
                memory_errors=sorted(set(memory_errors)),
            )
            stream.write(canonical(record).decode("utf-8") + "\n")
            stream.flush()
            records.append(record)
    report = aggregate(records)
    times = [r["seconds"] for r in records]
    report["latency_seconds"] = {
        key: float(np.percentile(times, percentile))
        for key, percentile in (("median", 50), ("p95", 95), ("p99", 99))
    }
    report["latency_seconds"]["mean"] = float(np.mean(times))
    report["by_split"] = {
        split: aggregate([r for r in records if r["split"] == split])
        for split in ("validation", "test")
    }
    report["paired_latency_seconds"] = (
        {
            variant: {
                name: float(
                    np.percentile(
                        [r["latency_seconds"][variant] for r in records if r["status"] == "ok"],
                        percentile,
                    )
                )
                for name, percentile in (("median", 50), ("p95", 95), ("p99", 99))
            }
            for variant in ("hybrid", "logistic")
        }
        if any(r["status"] == "ok" for r in records)
        else {}
    )
    report["latency_conditions"] = (
        "paired requests in a per-query session; alternating first method; includes cold startup"
    )
    report["memory_sampling_incomplete_queries"] = sum(bool(r["memory_errors"]) for r in records)
    report["memory_measurement"] = (
        "parent RSS after request and summed process-tree RSS at search phase boundaries; "
        "shared pages may be counted twice; not true peak memory"
    )
    report["sampled_process_tree_rss_max_bytes"] = max(
        r["sampled_process_tree_rss_max_bytes"] for r in records
    )
    report["rss_after_max_bytes"] = max(r["rss_after_bytes"] for r in records)
    report["regressions"] = [
        r["run_id"]
        for r in records
        if r["status"] == "ok" and any(r["logistic"][k] < r["hybrid"][k] for k in METRICS[:3])
    ]
    report["evaluation_scope"] = (
        "production retrieval, known explicit judgments; nonexhaustive qrels"
    )
    report["independent_holdout"] = dataset["independent_holdout"]
    return report
