"""Reproducible OCR-only comparison; production OCR and previous runs stay unchanged."""

import argparse
import json
import hashlib
import inspect
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

from benchmarks.execution import run_stage
from benchmarks.rapidocr_engine import RapidEngine, model_info
from benchmarks.resources import MemoryMonitor
from benchmarks.suite import (
    connect,
    image_quality,
    quality_status,
    save_stage,
    source_hash,
    summarize,
)
from realopen_lens.models import digest

ENGINES = ("tesseract", "rapidocr", "glm-ocr")
CORRECTIONS = Path(__file__).parent / "sources/evaluation-corrections-v1.json"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def inputs(dataset: Path) -> list[dict]:
    rows = json.loads((dataset / "manifest.json").read_text())["images"]
    edits = {r["id"]: r for r in json.loads(CORRECTIONS.read_text())["images"]}
    historical = json.loads((dataset / "manifest.json").read_text())["version"] == 2
    for row in rows:
        if digest(dataset / row["path"]) != row["sha256"]:
            raise ValueError(f"Image changed: {row['id']}")
        if historical and row["id"] in edits:
            edit = edits[row["id"]]
            if row.get("key_phrases") != edit["expected_key_phrases"]:
                raise ValueError(f"Reference correction precondition failed: {row['id']}")
            row["key_phrases"] = edit["key_phrases"]
            row["evaluation_correction"] = edit
    return rows


def worker(config_path: Path, engine: str) -> None:
    from realopen_lens.offline import enforce

    config = json.loads(config_path.read_text())
    if engine == "glm-ocr":
        enforce(loopback_port=config["glm_options"]["port"])
    else:
        enforce()
    db = connect(config_path.parent / "results.sqlite")
    done = {r[0] for r in db.execute("SELECT image_id FROM stages WHERE stage=?", (engine,))}
    pending = [r for r in config["images"] if r["id"] not in done]
    if not pending:
        db.close()
        return
    external_pids = tuple(config.get("ollama_server_pids", [])) if engine == "glm-ocr" else ()
    adapter = None
    with MemoryMonitor(external_pids=external_pids) as memory:
        start = perf_counter()
        if engine == "glm-ocr":
            from realopen_lens.ollama_ocr import OllamaOCR

            adapter = OllamaOCR(
                threads=config["threads"],
                **config["glm_options"],
                expected_digest=config["glm"]["digest"],
            )
            warmup = adapter.warmup()
            extract = adapter.extract
            environment = {
                **adapter.info,
                "warmup_metrics": warmup["ollama_metrics"],
                "loaded_model": adapter.loaded(),
                "memory_scope": (
                    "Python worker plus external Ollama server and descendants; "
                    "RSS does not fully measure Metal unified-memory allocation"
                ),
            }
        elif engine == "rapidocr":
            rapid = RapidEngine(config["threads"])
            extract = rapid.extract
            environment = {**rapid.info, "params": rapid.params}
        else:
            import shutil
            from realopen_lens.ocr import _environment, extract as tesseract

            environment = _environment(
                shutil.which("tesseract") or "tesseract", config["languages"]
            )

            def extract(path):
                return tesseract(path, config["languages"], config["threads"])

        environment["load_seconds"] = perf_counter() - start
        with db:
            db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                (engine + "_environment", json.dumps(environment)),
            )
        for i, row in enumerate(pending):
            memory.phase = row["id"]
            memory._sample()
            start = perf_counter()
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                    (engine + "_current", json.dumps({"id": row["id"], "started_at": monotonic()})),
                )
            try:
                result = {**extract(Path(config["dataset"]) / row["path"]), "status": "ok"}
            except Exception as error:
                result = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "warnings": [],
                }
            memory._sample()
            result.update(
                seconds=perf_counter() - start, peak_tree_rss_mib=memory.peaks[row["id"]] / 2**20
            )
            save_stage(db, row["id"], engine, result)
            if (i + 1) % 25 == 0:
                print(f"{engine}: {i + 1}/{len(pending)}", flush=True)
        memory.phase = "shutdown"
        if engine == "glm-ocr" and adapter is not None:
            try:
                adapter.close()
            except (OSError, ValueError) as error:
                with db:
                    db.execute(
                        "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                        (engine + "_cleanup_error", json.dumps(str(error))),
                    )
    with db:
        db.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?,?)",
            (
                engine + "_resources",
                json.dumps(
                    {
                        "peak_tree_rss_mib": max(memory.peaks.values()) / 2**20,
                        "samples": memory.samples,
                        "access_errors": memory.access_errors,
                        "sample_interval_ms": 20,
                    }
                ),
            ),
        )
    db.close()


def record(row: dict, result: dict) -> dict:
    expected = row["expected"]["decode"] == "reject"
    rejected = expected and "decoded pixels" in result.get("error", "")
    quality = image_quality(row, result.get("text", "")) if result["status"] == "ok" else {}
    verdict = ("pass" if rejected else "fail") if expected else quality_status(quality)
    if result["status"] != "ok" and not rejected:
        verdict = "fail"
    status = "expected_rejection" if rejected else result["status"]
    if status == "ok" and (result.get("warnings") or verdict == "fail"):
        status = "degraded"
    pixels = row["width"] * row["height"]
    return {
        **row,
        "status": status,
        "quality_status": verdict,
        "quality": quality,
        "actual": result,
        "responsible_stage": "ocr" if verdict == "fail" else None,
        "latency_seconds": result.get("seconds", 0),
        "peak_tree_rss_mib": result.get("peak_tree_rss_mib"),
        "resolution": (
            "over-24MP"
            if pixels > 24_000_000
            else "6-24MP" if pixels > 6_000_000 else "1-6MP" if pixels > 1_000_000 else "under-1MP"
        ),
    }


def report(output: Path, config: dict) -> dict:
    db = connect(output / "results.sqlite")
    summary: dict[str, Any] = {
        "completed_at": datetime.now(UTC).isoformat(),
        "config": config,
        "engines": {},
    }
    all_records = {}
    for engine in config.get("engines", ENGINES[:2]):
        payloads = {
            key: json.loads(value)
            for key, value in db.execute(
                "SELECT image_id,payload FROM stages WHERE stage=?", (engine,)
            )
        }
        if set(payloads) != {r["id"] for r in config["images"]}:
            raise ValueError(f"Incomplete {engine} run; resume instead of silently skipping")
        records = [record(row, payloads[row["id"]]) for row in config["images"]]
        all_records[engine] = {r["id"]: r for r in records}
        (output / f"{engine}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        metadata = {
            k: json.loads(v)
            for k, v in db.execute(
                "SELECT key,value FROM metadata WHERE key LIKE ?", (engine + "_%",)
            )
        }
        summary["engines"][engine] = {
            "aggregate": summarize(records),
            "metadata": metadata,
            "by_source": {
                kind: summarize([r for r in records if r["source"]["kind"] == kind])
                for kind in ("real", "synthetic", "derived")
                if any(r["source"]["kind"] == kind for r in records)
            },
        }
    for before_engine, after_engine in combinations(all_records, 2):
        changes = []
        for key, before in all_records[before_engine].items():
            after = all_records[after_engine][key]
            changes.append(
                {
                    "id": key,
                    "source_kind": before["source"]["kind"],
                    "category": before["category"],
                    "before": before["quality_status"],
                    "after": after["quality_status"],
                    before_engine + "_quality": before["quality"],
                    after_engine + "_quality": after["quality"],
                    before_engine + "_seconds": before["latency_seconds"],
                    after_engine + "_seconds": after["latency_seconds"],
                }
            )
        content = "".join(json.dumps(r) + "\n" for r in changes)
        (output / f"changes-{before_engine}-{after_engine}.jsonl").write_text(content)
        if (before_engine, after_engine) == ("tesseract", "rapidocr"):
            (output / "changes.jsonl").write_text(content)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    db.close()
    return summary


def merge_runs(sources: list[Path], output: Path) -> dict:
    """Combine disjoint engine runs only when inputs and evaluation are identical."""
    if output.exists():
        raise ValueError("Merged output must be a new directory")
    components = [(path, json.loads((path / "config.json").read_text())) for path in sources]
    if not components:
        raise ValueError("No component runs supplied")
    reference = components[0][1]
    selected: dict[str, Path] = {}
    for path, config in components:
        for key in (
            "manifest_sha256",
            "corrections_sha256",
            "images",
            "threads",
            "languages",
            "packages",
            "host",
            "evaluation_sha256",
            "image_decode_sha256",
        ):
            if key not in config or key not in reference:
                raise ValueError(f"Component run lacks identity field: {key}")
            if config[key] != reference[key]:
                raise ValueError(f"Component runs differ: {key}")
        for engine in config["engines"]:
            if engine in selected:
                raise ValueError(f"Duplicate engine component: {engine}")
            selected[engine] = path
    config = {
        **reference,
        "engines": [e for e in ENGINES if e in selected],
        "component_runs": {
            e: {"path": str(path), "config_sha256": digest(path / "config.json")}
            for e, path in selected.items()
        },
    }
    config.pop("source_sha256", None)
    for _, component in components:
        for key in ("glm", "glm_options", "ollama_server_pids", "rapidocr"):
            if component.get(key) is not None:
                config[key] = component[key]
    # Validate complete image coverage before creating any merged artifact.
    expected = {r["id"] for r in config["images"]}
    for engine, path in selected.items():
        source = connect(path / "results.sqlite")
        found = {
            r[0] for r in source.execute("SELECT image_id FROM stages WHERE stage=?", (engine,))
        }
        source.close()
        if found != expected:
            raise ValueError(f"Incomplete component run: {engine}")
    output.mkdir(parents=True)
    destination = connect(output / "results.sqlite")
    for engine, path in selected.items():
        source = connect(path / "results.sqlite")
        with destination:
            destination.executemany(
                "INSERT INTO stages VALUES (?,?,?)",
                source.execute(
                    "SELECT image_id,stage,payload FROM stages WHERE stage=?", (engine,)
                ),
            )
            destination.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                source.execute("SELECT key,value FROM metadata WHERE key LIKE ?", (engine + "_%",)),
            )
        source.close()
    destination.close()
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    return report(output, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmark-500-v2"))
    parser.add_argument("--output", type=Path, default=Path("runs/ocr-comparison"))
    parser.add_argument("--threads", type=int, choices=range(1, 33), default=2)
    parser.add_argument("--languages", default="eng+fra+ara")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=["tesseract"])
    parser.add_argument("--glm-model", default="glm-ocr:q8_0")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--ollama-server-pid", type=int)
    parser.add_argument("--glm-max-tokens", type=int, default=4096)
    parser.add_argument("--glm-context", type=int, default=8192)
    parser.add_argument("--glm-timeout", type=float, default=90)
    parser.add_argument("--worker", choices=ENGINES)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.merge:
        merge_runs(args.merge, args.output)
        return
    if args.worker:
        if args.config is None:
            parser.error("--worker needs --config")
        worker(args.config, args.worker)
        return
    if args.limit < 1:
        parser.error("--limit must be positive")
    from realopen_lens.inference import image_rgb

    evaluation = "".join(
        inspect.getsource(fn) for fn in (record, image_quality, quality_status, summarize)
    )
    evaluation += digest(Path(__file__).parent / "ocr_metrics.py")
    config = {
        "evaluation_sha256": hashlib.sha256(evaluation.encode()).hexdigest(),
        "image_decode_sha256": hashlib.sha256(inspect.getsource(image_rgb).encode()).hexdigest(),
        "dataset": str(args.dataset.resolve()),
        "manifest_sha256": digest(args.dataset / "manifest.json"),
        "corrections_sha256": digest(CORRECTIONS),
        "images": inputs(args.dataset)[: args.limit],
        "threads": args.threads,
        "languages": args.languages,
        "engines": list(dict.fromkeys(args.engines)),
        "rapidocr": model_info() if "rapidocr" in args.engines else None,
        "source_sha256": {
            "engine": source_hash(Path("src")),
            "harness": source_hash(Path("benchmarks")),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "rapidocr",
                "onnxruntime",
                "numpy",
                "Pillow",
                "opencv-python-headless",
                "python-bidi",
            )
        },
        "host": {"platform": platform.platform(), "python": sys.version},
        "limits": {
            "image_timeout": max(90, args.glm_timeout + 30),
            "startup_timeout": 180,
            "stage_timeout": 14400,
        },
    }
    if "glm-ocr" in args.engines:
        from realopen_lens.ollama_ocr import OllamaOCR
        from benchmarks.ollama_resources import server_pids

        options = {
            "model": args.glm_model,
            "port": args.ollama_port,
            "timeout": args.glm_timeout,
            "max_tokens": args.glm_max_tokens,
            "context": args.glm_context,
        }
        probe = OllamaOCR(threads=args.threads, **options)
        config["glm"] = probe.info
        config["glm_options"] = options
        config["ollama_server_pids"] = list(server_pids(args.ollama_server_pid))
        config["ollama_models_loaded_before_run"] = probe.request("/api/ps").get("models", [])
        if config["ollama_models_loaded_before_run"]:
            # A resumed interrupted GLM batch may retain our exact model. Never
            # evict unrelated models or silently benchmark under their memory load.
            if any(
                r.get("digest") != probe.info["digest"]
                for r in config["ollama_models_loaded_before_run"]
            ):
                raise ValueError("Other Ollama models are loaded; cannot isolate this benchmark")
        # Loaded state is recorded on first creation, not part of resumable identity.
        existing = args.output / "config.json"
        if existing.exists():
            config["ollama_models_loaded_before_run"] = json.loads(existing.read_text()).get(
                "ollama_models_loaded_before_run", []
            )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Run identity differs; use a new output directory")
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    db = connect(args.output / "results.sqlite")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path("src").resolve()) + os.pathsep + str(Path.cwd()),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "OMP_NUM_THREADS": str(args.threads),
        "VECLIB_MAXIMUM_THREADS": str(args.threads),
    }
    for engine in config["engines"]:
        print(f"Running {engine}; completed items are reused", flush=True)

        def failure(key: str, payload: dict) -> None:
            save_stage(db, key, engine, payload)

        events = run_stage(
            [
                sys.executable,
                "-m",
                "benchmarks.ocr_compare",
                "--worker",
                engine,
                "--config",
                str(path),
            ],
            db,
            engine,
            env,
            args.output / f"{engine}.log",
            failure,
            **config["limits"],
        )
        with (args.output / "events.jsonl").open("a") as stream:
            for event in events:
                stream.write(json.dumps(event) + "\n")
    db.close()
    result = report(args.output, config)
    for engine in config["engines"]:
        print(engine, json.dumps(result["engines"][engine]["aggregate"]["all"]["all"]), flush=True)


if __name__ == "__main__":
    main()
