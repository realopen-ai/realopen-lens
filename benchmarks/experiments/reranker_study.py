"""Predeclared feature/scope comparisons with fresh validation guards and frozen finalists."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load as load_judgments
from benchmarks.experiments.logistic import read_records, validate_records, quality, mean_metrics
from benchmarks.experiments.logistic_fit import fit
from benchmarks.reporting.collections import candidate_rows
from benchmarks.runtime.identity import source_hash
from benchmarks.runtime.resources import MemoryMonitor
from realopen_lens.learning.features import EXCLUDED
from realopen_lens.learning.logistic import load as load_model


def role(query: dict) -> str:
    return query.get("study_role", query["split"])


def aggregate(records: list[dict], queries: dict, model=None) -> tuple[dict, list]:
    groups: dict[str, list] = defaultdict(list)
    outputs = []
    for row in records:
        query = queries[row["query_id"]]
        ranking = (
            model.rank(row["trace"]["candidates"])[:50]
            if model is not None
            else row["trace"]["rankings"]["hybrid"]
        )
        metrics = quality(ranking, query)
        outputs.append({"query_id": query["id"], "ranking": ranking, "metrics": metrics})
        for tag in ("all", "language:" + query["language"], "kind:" + query["kind"]):
            groups[tag].append(metrics)
    return {
        tag: {"queries": len(values), **mean_metrics(values)} for tag, values in groups.items()
    }, outputs


def guard(candidate: dict, baseline: dict, specification: dict) -> list[str]:
    """Return every predeclared violation; no test metrics are accepted here."""
    epsilon = specification["tolerance"]
    violations = []
    for tag, values in baseline.items():
        if tag not in candidate:
            violations.append("missing-group:" + tag)
            continue
        if candidate[tag]["candidate_recall_at_50"] + epsilon < (
            values["candidate_recall_at_50"] - specification["max_recall_at_50_drop"]
        ):
            violations.append("recall:" + tag)
    for metric, minimum in (
        ("mrr_at_10", specification["min_mrr_at_10_delta"]),
        ("ndcg_at_10", specification["min_ndcg_at_10_delta_exclusive"]),
    ):
        delta = candidate["all"][metric] - baseline["all"][metric]
        if (metric == "ndcg_at_10" and delta <= minimum + epsilon) or delta < minimum - epsilon:
            violations.append(metric)
    return violations


def best(trials: list[dict]) -> dict:
    if not trials:
        raise ValueError("Cannot select from empty validation trials")
    return min(
        trials,
        key=lambda row: (
            -row["validation"]["all"]["ndcg_at_10"],
            -row["validation"]["all"]["mrr_at_10"],
            -row["l2"],
            row["profile"],
            row["scope"],
        ),
    )


def run(args) -> dict:
    corpus, bundle = load_judgments(args.corpus, args.judgments)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if any(item.get("protocol_sha256") != digest(args.protocol) for item in (corpus, bundle)):
        raise ValueError("Corpus/judgments must identify the predeclared protocol")
    if protocol["feature_profiles"] != {key: list(values) for key, values in EXCLUDED.items()}:
        raise ValueError("Protocol feature profiles differ from implementation")
    if protocol["rerank_scopes"] != ["union", "hybrid50"]:
        raise ValueError("Unsupported reranking scopes")
    records = read_records(args.baseline_dir / "queries.jsonl")
    config = json.loads((args.baseline_dir / "config.json").read_text(encoding="utf-8"))
    if config["corpus_sha256"] != digest(args.corpus) or config["judgments_sha256"] != digest(
        args.judgments
    ):
        raise ValueError("Production baseline uses different frozen inputs")
    validate_records(records, bundle, corpus, config)
    queries = {query["id"]: query for query in bundle["queries"]}
    if not config.get("defer_test_metrics") or any(
        role(queries[row["query_id"]]) == "test"
        and (not row.get("quality_deferred") or row.get("metrics") or "pool_recall" in row)
        for row in records
    ):
        raise ValueError("Fresh test quality must be deferred until validation selection is frozen")
    images = {row["sha256"]: row for row in corpus["images"]}
    if any(
        role(q) not in {"train", "validation", "test", "diagnostic"}
        or (role(q) != "diagnostic" and role(q) != q["split"])
        for q in queries.values()
    ):
        raise ValueError("Invalid study role or source split")
    legacy_groups = {
        images[label["image_sha256"]]["duplicate_group"]
        for query in queries.values()
        if query.get("source_exposure") != "fresh-v2"
        for label in query["judgments"]
        if label["grade"] > 0
    }
    if any(
        images[label["image_sha256"]]["duplicate_group"] in legacy_groups
        for query in queries.values()
        if role(query) in {"validation", "test"}
        for label in query["judgments"]
        if label["grade"] > 0
    ):
        raise ValueError("Fresh positive groups overlap earlier reviewed target groups")
    validation = [row for row in records if role(queries[row["query_id"]]) == "validation"]
    for fold in ("validation", "test"):
        subset = [q for q in queries.values() if role(q) == fold]
        if not {"documents", "screenshots", "photos"} <= {q["kind"] for q in subset} or {
            q["language"] for q in subset
        } != {"en", "fr", "ar"}:
            raise ValueError("Fresh held-out folds need all three languages and image kinds")
        if any(q.get("source_exposure") != "fresh-v2" or q["split"] != fold for q in subset):
            raise ValueError("Legacy diagnostic families cannot enter fresh selection or test")
    training = [
        candidate
        for row in records
        if role(queries[row["query_id"]]) == "train"
        for candidate in candidate_rows(row, queries[row["query_id"]], images)
        if candidate["training_eligible"]
    ]
    if args.output.exists():
        raise ValueError("Use a new output directory to preserve study history")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    identity = {
        "version": 1,
        "protocol_sha256": digest(args.protocol),
        "corpus_sha256": digest(args.corpus),
        "judgments_sha256": digest(args.judgments),
        "baseline_queries_sha256": digest(args.baseline_dir / "queries.jsonl"),
        "baseline_identity": config,
        "engine_sha256": source_hash(root / "src/realopen_lens"),
        "harness_sha256": source_hash(root / "benchmarks"),
    }
    save_json(args.output / "config.json", identity)
    for path in (args.protocol, args.corpus, args.judgments):
        shutil.copyfile(path, args.output / path.name)
    for source, target in (
        (root / "src/realopen_lens", "engine-source"),
        (root / "benchmarks", "harness-source"),
    ):
        shutil.copytree(source, args.output / target, ignore=shutil.ignore_patterns("__pycache__"))
    with (args.output / "training-train.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in training:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    spec = config["snapshot"]["pipeline_spec"]
    compatibility = {key: spec[key] for key in ("model_registry", "inference_sha256")}
    baseline, _ = aggregate(validation, queries)
    trials = []
    memory = MemoryMonitor()
    started = perf_counter()
    with (
        memory,
        (args.output / "validation-trials.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream,
    ):
        for profile in protocol["feature_profiles"]:
            for scope in protocol["rerank_scopes"]:
                for l2 in protocol["l2_grid"]:
                    artifact = fit(training, l2, compatibility, profile=profile, scope=scope)
                    artifact["provenance"] = {
                        key: value for key, value in identity.items() if key.endswith("sha256")
                    }
                    path = args.output / f"{profile}-{scope}-l2-{l2:g}.json"
                    save_json(path, artifact)
                    metrics, per_query = aggregate(validation, queries, load_model(path))
                    reasons = guard(metrics, baseline, protocol["validation_guard"])
                    trial = {
                        "profile": profile,
                        "scope": scope,
                        "l2": l2,
                        "validation": metrics,
                        "violations": reasons,
                        "model": path.name,
                    }
                    trials.append(trial)
                    for record in per_query:
                        stream.write(json.dumps({"trial": path.name, **record}) + "\n")
        guarded = [trial for trial in trials if not trial["violations"]]
        selected = best(guarded) if guarded else None
        champions = {
            scope: best([trial for trial in trials if trial["scope"] == scope])
            for scope in protocol["rerank_scopes"]
        }
        finalists = {}
        for scope, trial in champions.items():
            path = args.output / f"champion-{scope}.json"
            shutil.copyfile(args.output / trial["model"], path)
            finalists[scope] = load_model(path)
        if selected is not None:
            shutil.copyfile(args.output / selected["model"], args.output / "model.json")
            finalists["guarded"] = load_model(args.output / "model.json")
        if args.legacy_model is not None:
            shutil.copyfile(args.legacy_model, args.output / "legacy-model.json")
            finalists["legacy"] = load_model(args.output / "legacy-model.json")
        selection = {
            "validation_hybrid": baseline,
            "trials": trials,
            "selected": selected,
            "scope_champions": champions,
            "frozen_checksums": {name: model.sha256 for name, model in finalists.items()},
        }
        save_json(args.output / "selection.json", selection)
        # Fresh test metrics are calculated only after all finalists have been frozen.
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        pool_groups: dict[str, list[float]] = defaultdict(list)
        durations: dict[str, list] = defaultdict(list)
        with (args.output / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in records:
                query = queries[row["query_id"]]
                candidates = row["trace"]["candidates"]
                rankings = {"hybrid": row["trace"]["rankings"]["hybrid"]}
                scores = {}
                for name, model in finalists.items():
                    ranking = model.rank(candidates)
                    rankings[name] = ranking
                    samples = []
                    for _ in range(10):
                        start = perf_counter()
                        model.rank(candidates)
                        samples.append((perf_counter() - start) * 1000)
                    durations[name].append(float(np.median(samples)))
                    scores[name] = model.scores(candidates).tolist()
                metrics = {name: quality(ranking[:50], query) for name, ranking in rankings.items()}
                positives = {r["image_sha256"] for r in query["judgments"] if r["grade"] > 0}
                missed = sorted(positives - {r["image_id"] for r in candidates})
                pool_recall = (len(positives) - len(missed)) / len(positives)
                for tag in (
                    "role:" + role(query),
                    "role:" + role(query) + "/kind:" + query["kind"],
                    "role:" + role(query) + "/language:" + query["language"],
                ):
                    pool_groups[tag].append(pool_recall)
                    for name, values in metrics.items():
                        groups[tag][name].append(values)
                stream.write(
                    json.dumps(
                        {
                            "query_id": query["id"],
                            "query": query,
                            "rankings": rankings,
                            "metrics": metrics,
                            "candidate_ids": [r["image_id"] for r in candidates],
                            "scores": scores,
                            "pool_recall": pool_recall,
                            "missed_pool_positives": missed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    from benchmarks.metrics.statistics import distribution

    summary = {
        "version": 1,
        "queries": len(records),
        "training_pairs": len(training),
        "trials": len(trials),
        "guarded_trials": len(guarded),
        "selection": selected,
        "hybrid_remains_default": True,
        "known_positive_pool_recall": {
            tag: {"queries": len(values), "mean": float(np.mean(values))}
            for tag, values in pool_groups.items()
        },
        "groups": {
            tag: {
                mode: {"queries": len(values), **mean_metrics(values)}
                for mode, values in modes.items()
            }
            for tag, modes in groups.items()
        },
        "rerank_only_ms": {name: distribution(values) for name, values in durations.items()},
        "fit_and_replay_seconds": perf_counter() - started,
        "sampled_process_tree_peak_mib": max(memory.peaks.values(), default=0) / 2**20,
    }
    save_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                "queries": len(records),
                "guarded_trials": len(guarded),
                "selected": selected["model"] if selected else "hybrid",
            }
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-dir", "corpus", "judgments", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
