"""Resumable per-image evaluation with immutable run identities and retained evidence."""

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import shutil
import sys
from time import perf_counter
from collections import Counter, defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from benchmarks.dataset import digest
from benchmarks.ocr_metrics import evaluate, normalized
from benchmarks.retrieval_metrics import metrics
from benchmarks.resources import MemoryMonitor
from benchmarks.execution import run_stage
from realopen_lens.models import registry, verify


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS stages (
            image_id TEXT, stage TEXT, payload TEXT NOT NULL,
            PRIMARY KEY(image_id, stage));
        CREATE TABLE IF NOT EXISTS stage_history (
            attempt INTEGER PRIMARY KEY, image_id TEXT, stage TEXT, payload TEXT,
            archived_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS vectors (
            image_id TEXT, stage TEXT, shape TEXT, data BLOB,
            PRIMARY KEY(image_id, stage));
    """)
    return db


def source_hash(root: Path) -> str:
    value = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def save_stage(
    db: sqlite3.Connection,
    image_id: str,
    stage: str,
    payload: dict,
    vector: np.ndarray | None = None,
) -> None:
    with db:
        db.execute(
            "INSERT INTO stage_history(image_id,stage,payload) "
            "SELECT image_id,stage,payload FROM stages WHERE image_id=? AND stage=?",
            (image_id, stage),
        )
        db.execute("DELETE FROM vectors WHERE image_id=? AND stage=?", (image_id, stage))
        if payload.get("status") != "ok":
            vector = None
        if vector is not None:
            vector = np.asarray(vector, dtype=np.float32)
            db.execute(
                "INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                (image_id, stage, json.dumps(vector.shape), vector.tobytes()),
            )
        db.execute(
            "INSERT OR REPLACE INTO stages VALUES (?,?,?)",
            (image_id, stage, json.dumps(payload, ensure_ascii=False)),
        )


def reset_errors(db: sqlite3.Connection) -> None:
    with db:
        db.execute(
            "INSERT INTO stage_history(image_id,stage,payload) "
            "SELECT image_id,stage,payload FROM stages WHERE "
            "json_extract(payload,'$.status') IN ('error','timeout','blocked')"
        )
        db.execute(
            "DELETE FROM vectors WHERE EXISTS (SELECT 1 FROM stages "
            "WHERE stages.image_id=vectors.image_id AND stages.stage=vectors.stage "
            "AND json_extract(payload,'$.status') IN ('error','timeout','blocked'))"
        )
        db.execute(
            "DELETE FROM stages WHERE json_extract(payload,'$.status') "
            "IN ('error','timeout','blocked')"
        )


def load_vectors(db: sqlite3.Connection, stage: str) -> tuple[list[str], np.ndarray]:
    rows = db.execute(
        "SELECT image_id,shape,data FROM vectors WHERE stage=? ORDER BY image_id", (stage,)
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    ids = []
    arrays = []
    for image_id, shape, data in rows:
        array = np.frombuffer(data, dtype=np.float32).reshape(json.loads(shape))
        arrays.append(array)
        ids.extend([image_id] * len(array))
    return ids, np.concatenate(arrays)


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        **{
            name: float(np.percentile(values, p))
            for name, p in [("median", 50), ("p95", 95), ("p99", 99)]
        },
        "max": float(max(values)),
    }


def summarize(records: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for field in ["all", "category", "subcategory", "format", "split", "difficulty", "resolution"]:
        grouped: dict[str, list] = defaultdict(list)
        for item in records:
            labels = (
                ["all"]
                if field == "all"
                else item.get(field, []) if field == "difficulty" else [item.get(field, "unknown")]
            )
            for label in labels:
                grouped[str(label)].append(item)
        result[field] = {}
        for label, items in sorted(grouped.items()):
            counts = Counter(r["status"] for r in items)
            evaluated = [r for r in items if r["quality_status"] != "not_evaluated"]
            quality_items = [row["quality"] for row in items if row.get("quality")]
            totals = {
                key: sum(row[key] for row in quality_items)
                for key in (
                    "character_errors",
                    "reference_characters",
                    "word_errors",
                    "reference_words",
                    "matched_tokens",
                    "reference_tokens",
                    "observed_tokens",
                    "matched_key_phrases",
                    "reference_key_phrases",
                )
            }
            quality_metrics = {
                name: totals[num] / totals[den] if totals[den] else None
                for name, num, den in (
                    ("cer", "character_errors", "reference_characters"),
                    ("wer", "word_errors", "reference_words"),
                    ("token_recall", "matched_tokens", "reference_tokens"),
                    ("token_precision", "matched_tokens", "observed_tokens"),
                    ("key_phrase_recall", "matched_key_phrases", "reference_key_phrases"),
                )
            }
            result[field][label] = {
                "ocr": {"evaluated_images": len(quality_items), **totals, **quality_metrics},
                "count": len(items),
                "statuses": dict(counts),
                "execution_success_rate": sum(
                    r["status"] not in ("error", "timeout") for r in items
                )
                / len(items),
                "quality_evaluated": len(evaluated),
                "quality_pass_rate": (
                    sum(r["quality_status"] == "pass" for r in evaluated) / len(evaluated)
                    if evaluated
                    else None
                ),
                "latency_seconds": distribution([r["latency_seconds"] for r in items]),
                "peak_tree_rss_mib": distribution(
                    [r["peak_tree_rss_mib"] for r in items if r["peak_tree_rss_mib"] is not None]
                ),
                "timeout_rate": counts["timeout"] / len(items),
            }
    return result


def query_items(rows: list[dict]) -> list[dict]:
    queries = []
    for row in rows:
        if row["source"]["kind"] == "derived" or row["expected"]["decode"] != "success":
            continue
        text = row.get("query")
        if not text and row.get("key_phrases"):
            text = " ".join(row["key_phrases"][:2])
        if text:
            # Templates share split groups; relevance includes only the same content.
            relevant = [
                r["id"]
                for r in rows
                if (r["id"] == row["id"] or r["source"].get("parent") == row["id"])
                and r["expected"]["decode"] == "success"
                and r["subcategory"] != "crop"
            ]
            variants = [{"language": row.get("language", "en"), "text": text, "kind": "natural"}]
            if row.get("identifier_query"):
                variants.append(
                    {
                        "language": row.get("language", "en"),
                        "text": row["identifier_query"],
                        "kind": "identifier",
                    }
                )
            variants.extend(
                {**item, "kind": "translated"} for item in row.get("query_variants", [])
            )
            seen = set()
            for number, item in enumerate(variants):
                if item["text"] in seen:
                    continue
                seen.add(item["text"])
                queries.append(
                    {
                        "id": row["id"] + f"::q{number}",
                        "text": item["text"],
                        "language": item["language"],
                        "kind": item["kind"],
                        "category": row["category"],
                        "subcategory": row["subcategory"],
                        "split": row["split"],
                        "relevant": relevant,
                        "judgment": (
                            "Published caption or source/visually authored known positive; "
                            "translations not independently reviewed"
                        ),
                    }
                )
    return queries


def image_quality(row: dict, text: str) -> dict:
    if "truth" in row:
        return evaluate(
            [{"image_id": row["id"], "text": text}],
            [
                {
                    "image_id": row["id"],
                    "text": row["truth"],
                    "language": row.get("language", "en"),
                    "key_phrases": row.get("key_phrases", []),
                }
            ],
        )["all"]
    phrases = row.get("key_phrases", [])
    if not phrases:
        return {}
    empty = evaluate(
        [{"image_id": row["id"], "text": ""}],
        [{"image_id": row["id"], "text": "", "language": "en"}],
    )["all"]
    return {
        **empty,
        "matched_key_phrases": sum(normalized(p) in normalized(text) for p in phrases),
        "reference_key_phrases": len(phrases),
        "key_phrase_recall": sum(normalized(p) in normalized(text) for p in phrases) / len(phrases),
    }


def quality_status(quality: dict) -> str:
    if not quality:
        return "not_evaluated"
    checks = [
        quality[key] >= 0.8
        for key in ("token_recall", "token_precision", "key_phrase_recall")
        if quality.get(key) is not None
    ]
    return "pass" if checks and all(checks) else "fail"


def report(output: Path, config: dict) -> dict:
    db = connect(output / "results.sqlite")
    rows = config["images"]
    records: list[dict[str, Any]] = []
    stage_lookup = {
        (key, stage): json.loads(payload)
        for key, stage, payload in db.execute("SELECT image_id,stage,payload FROM stages")
    }
    for row in rows:
        stages = {
            stage: stage_lookup.get((row["id"], stage), {"status": "missing"})
            for stage in ("ocr", "vision", "semantic")
        }
        statuses = [r["status"] for r in stages.values()]
        expected_reject = row["expected"]["decode"] == "reject"
        failures = [s for s, p in stages.items() if p["status"] in ("error", "timeout", "missing")]
        rejected = expected_reject and "decoded pixels" in stages["vision"].get("error", "")
        status = (
            "expected_rejection"
            if rejected
            else (
                "timeout"
                if "timeout" in statuses
                else (
                    "error"
                    if failures
                    else "degraded" if any(p.get("warnings") for p in stages.values()) else "ok"
                )
            )
        )
        quality = (
            image_quality(row, stages["ocr"].get("text", ""))
            if stages["ocr"]["status"] == "ok"
            else {}
        )
        quality_verdict = (
            ("pass" if rejected else "fail") if expected_reject else quality_status(quality)
        )
        if quality_verdict == "fail" and status == "ok":
            status = "degraded"
        if failures and not rejected:
            quality_verdict = "fail"
        pixels = row["width"] * row["height"]
        record = {
            **row,
            "status": status,
            "quality_status": quality_verdict,
            "quality": quality,
            "stages": stages,
            "responsible_stages": failures or (["ocr"] if quality_verdict == "fail" else []),
            "resolution": (
                "over-24MP"
                if pixels > 24_000_000
                else (
                    "6-24MP"
                    if pixels > 6_000_000
                    else "1-6MP" if pixels > 1_000_000 else "under-1MP"
                )
            ),
            "latency_seconds": sum(p.get("seconds", 0) for p in stages.values()),
            "peak_tree_rss_mib": max(
                (p.get("peak_tree_rss_mib") or 0 for p in stages.values()), default=0
            ),
        }
        records.append(record)
    with (output / "individual.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    retrieval = retrieval_report(db, config, output)
    summary: dict[str, Any] = {
        "config": config,
        "completed_at": datetime.now(UTC).isoformat(),
        "images": len(rows),
        "aggregate": summarize(records),
        "retrieval": retrieval,
        "limitations": [
            "OCR diagnostics require >=80% token recall/precision and key-phrase recall; "
            "CER/WER remain order-sensitive diagnostics",
            "Untranscribed images have no quality verdict; execution is not accuracy",
            "RSS includes resident models and children; sampled, not incremental",
            "Stage sum latency excludes model loading and retrieval; consult wall times",
            "Relevance judgments are known positives and incomplete",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    failure_records = [
        {
            "id": r["id"],
            "category": r["category"],
            "split": r["split"],
            "status": r["status"],
            "responsible_stages": r["responsible_stages"],
            "quality": r["quality"],
            "latency_seconds": r["latency_seconds"],
            "image": str(Path(config["dataset"]) / r["path"]),
            "evidence": "individual.jsonl and results.sqlite: OCR, boxes, confidence, errors",
        }
        for r in records
        if r["status"] != "ok" or r["quality_status"] == "fail"
    ]
    (output / "failures.json").write_text(
        json.dumps(failure_records, ensure_ascii=False, indent=2) + "\n"
    )
    db.close()
    print(json.dumps(summary["aggregate"]["all"], indent=2), flush=True)
    return summary


def retrieval_report(db: sqlite3.Connection, config: dict, output: Path) -> dict:
    from realopen_lens.retrieval import fuse, lexical_index, rank_lexical, rank_vectors

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
                from realopen_lens.retrieval import hybrid_rank, lexical_frequencies

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


def write_resources(output: Path, memory: MemoryMonitor) -> None:
    current: dict[str, Any] = {
        "peak_tree_rss_mib": max(memory.peaks.values(), default=0) / 2**20,
        "phase_peak_mib": {key: value / 2**20 for key, value in memory.peaks.items()},
        "samples": memory.samples,
        "access_errors": memory.access_errors,
        "sample_interval_ms": 20,
        "scope": "Coordinator and children, including model loading and evaluation",
        "limitations": "Shared pages may count twice; spikes between samples may be missed",
    }
    number = len(list(output.glob("resources-attempt-*.json"))) + 1
    attempt = output / f"resources-attempt-{number:03d}.json"
    attempt.write_text(json.dumps(current, indent=2) + "\n")
    destination = output / "resources.json"
    previous = json.loads(destination.read_text()) if destination.exists() else {}
    peaks = previous.get("phase_peak_mib", {})
    for key, value in current["phase_peak_mib"].items():
        peaks[key] = max(peaks.get(key, 0), value)
    merged = {
        **current,
        "phase_peak_mib": peaks,
        "peak_tree_rss_mib": max(
            current["peak_tree_rss_mib"], previous.get("peak_tree_rss_mib", 0)
        ),
        "access_errors": current["access_errors"] + previous.get("access_errors", 0),
        "samples": current["samples"] + previous.get("samples", 0),
        "invocations": previous.get("invocations", []) + [attempt.name],
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=2) + "\n")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmark-500"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, default=Path("src"))
    parser.add_argument("--variant", choices=["baseline", "improved"], required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--languages", default="eng+fra+ara")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-timeout", type=float, default=90)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    manifest = args.dataset / "manifest.json"
    rows = json.loads(manifest.read_text())["images"]
    if args.image_timeout <= 0 or args.startup_timeout <= 0:
        raise ValueError("Worker timeouts must be positive")
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            raise ValueError("Limit must select at least one available image")
        rows = rows[: args.limit]
    for row in rows:
        path = (args.dataset / row["path"]).resolve()
        if not path.is_relative_to(args.dataset.resolve()) or digest(path) != row["sha256"]:
            raise ValueError(f"Invalid dataset input: {row['id']}")
    verify(args.models, "int8")
    identity = {
        "dataset": str(args.dataset.resolve()),
        "manifest_sha256": digest(manifest),
        "images": rows,
        "queries": query_items(rows),
        "engine_root": str(args.engine_root.resolve()),
        "engine_sha256": source_hash(args.engine_root),
        "variant": args.variant,
        "models": str(args.models.resolve()),
        "artifacts": registry(),
        "threads": args.threads,
        "languages": args.languages,
        "worker_limits": {
            "image_timeout": args.image_timeout,
            "startup_timeout": args.startup_timeout,
        },
        "runner_sha256": source_hash(Path("benchmarks")),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "ram_gib": psutil.virtual_memory().total / 2**30,
            "python": sys.version,
        },
        "packages": {
            name: version(name)
            for name in ["numpy", "Pillow", "onnxruntime", "tokenizers", "psutil", "realopen-lens"]
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError(
            "Resume configuration/source/environment mismatch; choose a new output directory"
        )
    config_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")
    for source, name in (
        (args.engine_root, "engine-source"),
        (Path("benchmarks"), "harness-source"),
    ):
        if not (args.output / name).exists():
            shutil.copytree(
                source, args.output / name, ignore=shutil.ignore_patterns("__pycache__")
            )
    db = connect(args.output / "results.sqlite")
    if args.retry_errors:
        reset_errors(db)
    db.close()
    memory = MemoryMonitor()
    try:
        with memory:
            for stage in ["ocr", "vision", "semantic", "visual_queries", "semantic_queries"]:
                if source_hash(args.engine_root) != identity["engine_sha256"]:
                    raise ValueError(
                        "Engine changed during run; resume with preserved original source"
                    )
                memory.phase = stage
                print(f"Running {stage} (completed records are reused)", flush=True)
                env = {
                    **os.environ,
                    "PYTHONPATH": str(args.engine_root.resolve()),
                    "OMP_NUM_THREADS": str(args.threads),
                    "OPENBLAS_NUM_THREADS": str(args.threads),
                    "TOKENIZERS_PARALLELISM": "false",
                }
                stage_start = perf_counter()
                stage_db = connect(args.output / "results.sqlite")
                try:
                    events = run_stage(
                        [
                            sys.executable,
                            "-m",
                            "benchmarks.suite_worker",
                            str(config_path.resolve()),
                            stage,
                        ],
                        stage_db,
                        stage,
                        env,
                        args.output / f"{stage}.log",
                        lambda image_id, payload: save_stage(stage_db, image_id, stage, payload),
                        image_timeout=args.image_timeout,
                        startup_timeout=args.startup_timeout,
                        stage_timeout=max(600, len(rows) * 60),
                    )
                finally:
                    stage_db.close()
                with (args.output / "run-events.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "stage": stage,
                                "seconds": perf_counter() - stage_start,
                                "returncode": 0,
                                "supervision": events,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                        )
                        + "\n"
                    )
            memory.phase = "evaluation"
            report(args.output, identity)
    finally:
        write_resources(args.output, memory)


if __name__ == "__main__":
    main()
