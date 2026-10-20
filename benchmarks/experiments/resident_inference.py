"""Compare paired one-shot and resident production search, with retained request profiles."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
from time import perf_counter

import psutil

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.metrics.collection import evaluate
from benchmarks.metrics.statistics import distribution
from benchmarks.runtime.collections import snapshot
from benchmarks.runtime.identity import source_hash
from benchmarks.runtime.resources import MemoryMonitor
from realopen_lens.search import SearchSession, search_collection


def select_queries(bundle: dict, families_per_kind: int, seed: int) -> list[dict]:
    if families_per_kind < 1:
        raise ValueError("Select at least one family per kind")
    groups: dict[str, set] = {}
    for query in bundle["queries"]:
        groups.setdefault(query.get("kind", "mixed"), set()).add(query["family"])
    rng = random.Random(seed)
    selected = set()
    for kind in sorted(groups):
        families = sorted(groups[kind])
        if len(families) < families_per_kind:
            raise ValueError("Not enough reviewed families for the requested cohort")
        selected.update(rng.sample(families, families_per_kind))
    return sorted((q for q in bundle["queries"] if q["family"] in selected), key=lambda q: q["id"])


def host() -> dict:
    battery = psutil.sensors_battery()
    return {
        "platform": platform.platform(),
        "memory_gib": psutil.virtual_memory().total / 2**30,
        "available_gib": psutil.virtual_memory().available / 2**30,
        "swap_used_gib": psutil.swap_memory().used / 2**30,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "power_plugged": battery.power_plugged if battery else None,
    }


def measure(
    query: dict,
    index: Path,
    collection: str,
    models: Path,
    threads: int,
    session: SearchSession | None = None,
) -> dict:
    memory = MemoryMonitor()
    profile: dict = {}
    trace: dict = {}
    started = perf_counter()
    result = {"query_id": query["id"], "query": query, "host_before": host()}
    with memory:
        try:
            if session is None:
                found = search_collection(
                    query["text"],
                    index_dir=index,
                    collection=collection,
                    models=models,
                    threads=threads,
                    limit=50,
                    trace=trace,
                    profile=profile,
                )
            else:
                found = session.search(query["text"], limit=50, trace=trace, profile=profile)
            labels = {j["image_sha256"]: j["grade"] for j in query["judgments"]}
            result.update(
                status="ok",
                results=[asdict(r) for r in found],
                trace=trace,
                metrics=evaluate([r.image_id for r in found], labels),
            )
        except Exception as error:
            result.update(status="error", error_type=type(error).__name__, error=str(error))
    result.update(
        seconds=perf_counter() - started,
        profile=profile,
        peak_tree_rss_mib=max(memory.peaks.values(), default=0) / 2**20,
        memory_samples=memory.samples,
        memory_access_errors=memory.access_errors,
        worker_pid=session.worker_pid if session else None,
        host_after=host(),
    )
    result["resident_worker_rss_mib"] = None
    if session is not None and session.worker_pid is not None:
        try:
            result["resident_worker_rss_mib"] = (
                psutil.Process(session.worker_pid).memory_info().rss / 2**20
            )
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError):
            result["memory_access_errors"] += 1
    return result


def summarize(records: list[dict]) -> dict:
    groups = {}
    for policy in ("int8", "fp32"):
        for phase in ("oneshot", "resident_cold", "resident_warm"):
            rows = [r for r in records if r["policy"] == policy and r["phase"] == phase]
            ok = [r for r in rows if r["status"] == "ok"]
            groups[policy + "/" + phase] = {
                "requests": len(rows),
                "errors": len(rows) - len(ok),
                "seconds": distribution([r["seconds"] for r in rows]),
                "peak_tree_rss_mib": distribution([r["peak_tree_rss_mib"] for r in rows]),
                "memory_access_errors": sum(r["memory_access_errors"] for r in rows),
                "worker_pids": sorted(
                    {r["worker_pid"] for r in rows if r["worker_pid"] is not None}
                ),
                "resident_worker_rss_mib": distribution(
                    [
                        r["resident_worker_rss_mib"]
                        for r in rows
                        if r["resident_worker_rss_mib"] is not None
                    ]
                ),
                "profiles": {
                    key: distribution([r["profile"][key] for r in ok if key in r["profile"]])
                    for key in sorted({key for r in ok for key in r["profile"]})
                },
                "quality": (
                    {key: sum(r["metrics"][key] for r in ok) / len(ok) for key in ok[0]["metrics"]}
                    if ok
                    else {}
                ),
            }
    return groups


def run(args) -> dict:
    corpus, bundle = load(args.corpus, args.judgments)
    queries = select_queries(bundle, args.families_per_kind, args.seed)
    states = {
        policy: snapshot(
            index,
            args.collection,
            corpus,
            corpus_prefix=args.corpus_prefix,
            allow_extra_images=args.allow_extra_images,
        )
        for policy, index in (("int8", args.int8_index), ("fp32", args.fp32_index))
    }
    for policy, state in states.items():
        if state["pipeline_spec"].get("siglip_precision", "int8") != policy:
            raise ValueError("Index precision does not match its policy")
    identity = {
        "version": 1,
        "seed": args.seed,
        "queries": queries,
        "snapshots": states,
        "index_dirs": {
            "int8": str(args.int8_index.resolve()),
            "fp32": str(args.fp32_index.resolve()),
        },
        "models": str(args.models.resolve()),
        "threads": args.threads,
        "repeats": args.repeats,
        "corpus_sha256": digest(args.corpus),
        "judgments_sha256": digest(args.judgments),
        "engine_sha256": source_hash(Path("src/realopen_lens")),
        "harness_sha256": source_hash(Path("benchmarks")),
        "packages": {
            name: version(name) for name in ("numpy", "onnxruntime", "tokenizers", "psutil")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "config.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != identity:
        raise ValueError("Run identity changed; choose a new output directory")
    save_json(config, identity)
    for source, target in (
        (Path("src/realopen_lens"), "engine-source"),
        (Path("benchmarks"), "harness-source"),
    ):
        if not (args.output / target).exists():
            shutil.copytree(
                source, args.output / target, ignore=shutil.ignore_patterns("__pycache__")
            )
    with sqlite3.connect(args.output / "results.sqlite") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY,payload TEXT NOT NULL)"
        )

        def retain(policy: str, phase: str, repeat: int, query: dict, index: Path, session=None):
            key = f"{policy}/{phase}/{repeat}/{query['id']}"
            if (
                phase == "oneshot"
                and db.execute("SELECT 1 FROM requests WHERE key=?", (key,)).fetchone()
            ):
                return
            # Warm requests are never resumed in a newly cold worker. A complete resident
            # attempt is retained before rerunning the whole sequence after interruption.
            row = measure(query, index, args.collection, args.models, args.threads, session)
            row.update(policy=policy, phase=phase, repeat=repeat)
            with db:
                db.execute("INSERT OR REPLACE INTO requests VALUES (?, ?)", (key, json.dumps(row)))
            print(key, row["status"], f"{row['seconds']:.3f}s", flush=True)

        for policy, index in (("int8", args.int8_index), ("fp32", args.fp32_index)):
            for query in queries:
                retain(policy, "oneshot", 0, query, index)
            count = db.execute(
                "SELECT COUNT(*) FROM requests WHERE key LIKE ?", (policy + "/resident_%",)
            ).fetchone()[0]
            expected = 1 + args.repeats * len(queries)
            if count != expected:
                old = [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT payload FROM requests WHERE key LIKE ?", (policy + "/resident_%",)
                    )
                ]
                if old:
                    number = len(list(args.output.glob(f"{policy}-interrupted-*.json"))) + 1
                    save_json(
                        args.output / f"{policy}-interrupted-{number:03}.json", {"requests": old}
                    )
                    with db:
                        db.execute(
                            "DELETE FROM requests WHERE key LIKE ?", (policy + "/resident_%",)
                        )
                with SearchSession(
                    index_dir=index,
                    collection=args.collection,
                    models=args.models,
                    threads=args.threads,
                ) as session:
                    retain(policy, "resident_cold", 0, queries[0], index, session)
                    for repeat in range(args.repeats):
                        for query in queries:
                            retain(policy, "resident_warm", repeat, query, index, session)
        for policy, index in (("int8", args.int8_index), ("fp32", args.fp32_index)):
            if (
                snapshot(
                    index,
                    args.collection,
                    corpus,
                    corpus_prefix=args.corpus_prefix,
                    allow_extra_images=args.allow_extra_images,
                )
                != states[policy]
            ):
                raise ValueError("Collection changed during evaluation")
        records = [
            json.loads(r[0]) for r in db.execute("SELECT payload FROM requests ORDER BY key")
        ]
    baseline = {(r["policy"], r["query_id"]): r for r in records if r["phase"] == "oneshot"}
    pairs = []
    for row in records:
        if row["phase"] == "oneshot":
            continue
        other = baseline[(row["policy"], row["query_id"])]
        ok = row["status"] == other["status"] == "ok"
        pairs.append(
            {
                "policy": row["policy"],
                "phase": row["phase"],
                "repeat": row["repeat"],
                "query_id": row["query_id"],
                "both_successful": ok,
                "identical_trace": ok and row["trace"] == other["trace"],
                "identical_results": ok and row["results"] == other["results"],
            }
        )
    report = {
        "identity": identity,
        "summary": summarize(records),
        "pairs": pairs,
        "all_pairs_identical": all(r["identical_trace"] and r["identical_results"] for r in pairs),
        "errors": sum(r["status"] != "ok" for r in records),
        "limitations": (
            "Diagnostic fixed multilingual subset; OS caches and background load are uncontrolled. "
            "Resident cold is one observed startup per policy, not a latency distribution. "
            "RSS includes coordinator and worker; shared pages can count twice."
        ),
    }
    save_json(args.output / "comparison.json", report)
    with (args.output / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int8-index", type=Path, required=True)
    parser.add_argument("--fp32-index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--corpus-prefix", type=Path, default=Path("."))
    parser.add_argument("--allow-extra-images", action="store_true")
    parser.add_argument("--collection", default="personal")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--families-per-kind", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = run(args)
    if report["errors"] or not report["all_pairs_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
