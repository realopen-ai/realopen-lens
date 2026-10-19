"""Reproducible OCR-only comparison; production OCR and previous runs stay unchanged."""

import argparse
import json
import hashlib
import inspect
import os
import platform
import sys
from pathlib import Path
from time import monotonic, perf_counter
from benchmarks.runtime.execution import run_stage
from benchmarks.experiments.rapidocr_engine import RapidEngine, model_info
from benchmarks.runtime.resources import MemoryMonitor
from benchmarks.storage.results import connect, save_stage
from benchmarks.metrics.quality import image_quality, quality_status
from benchmarks.runtime.identity import source_hash
from benchmarks.metrics.statistics import summarize
from realopen_lens.models.manager import digest
from benchmarks.runtime.identity import package_version
from benchmarks.datasets.references import inputs
from benchmarks.metrics.records import record
from benchmarks.reporting.ocr import report, merge_runs
from benchmarks.datasets.references import CORRECTIONS
from benchmarks.runtime.identity import ENGINES
from benchmarks.metrics import ocr as ocr_metrics


def worker(config_path: Path, engine: str) -> None:
    from realopen_lens.runtime.offline import enforce

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
            from realopen_lens.ocr.ollama import OllamaOCR

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
            from realopen_lens.ocr.tesseract import _environment, extract as tesseract

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
    from realopen_lens.encoders.onnx import image_rgb

    evaluation = "".join(
        inspect.getsource(fn) for fn in (record, image_quality, quality_status, summarize)
    )
    evaluation += digest(Path(ocr_metrics.__file__))
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
        from realopen_lens.ocr.ollama import OllamaOCR
        from benchmarks.runtime.ollama import server_pids

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
                "benchmarks.experiments.ocr_compare",
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
