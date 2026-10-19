"""Reproducible, local-only benchmark orchestration and private reports."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
import psutil
from benchmarks.datasets.judgments import queries
from benchmarks.metrics.ocr import evaluate as evaluate_ocr
from benchmarks.metrics.ocr import by_category as ocr_by_category
from benchmarks.metrics.ocr import load_truth
from benchmarks.runtime.resources import MemoryMonitor
from realopen_lens.indexing.scanner import scan
from realopen_lens.models.manager import digest, registry, verify
from benchmarks.reporting.baseline import evaluate, markdown


def run(
    data: Path,
    models: Path,
    output: Path,
    query_file: Path | None,
    precision: str,
    languages: str,
    limit: int,
    batch_size: int,
    threads: int,
    repeats: int,
    corpus_kind: str,
    progress: Callable[[str], None] = print,
    ocr_truth: Path | None = None,
) -> Path:
    image_ids = scan(data.resolve(), limit)
    query_items = queries(query_file, image_ids)
    transcriptions = load_truth(ocr_truth, image_ids)
    if output.exists():
        raise ValueError(f"Output directory already exists; choose a new run directory: {output}")
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise ValueError("Tesseract 5 is required. See docs/benchmark.md for native installation.")
    progress("Verifying local model checksums (no downloads)")
    verify(models, precision)
    output.mkdir(parents=True)
    config = {
        "data": str(data.resolve()),
        "models": str(models.resolve()),
        "output": str(output.resolve()),
        "images": image_ids,
        "queries": query_items,
        "precision": precision,
        "languages": languages,
        "batch_size": batch_size,
        "threads": threads,
        "repeats": repeats,
        "tesseract": tesseract,
        "ocr_truth": transcriptions,
    }
    config_file = output / "config.json"
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    stages = {}
    pipeline_start = perf_counter()
    with MemoryMonitor() as memory:
        for stage in ("ocr", "vision", "visual_queries", "semantic"):
            if stage == "visual_queries" and not query_items:
                continue
            memory.phase = stage
            progress(f"Running {stage}")
            start = perf_counter()
            with (output / f"{stage}.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.runtime.baseline_worker",
                        stage,
                        str(config_file.resolve()),
                    ],
                    stdout=log,
                    stderr=log,
                    env={
                        **os.environ,
                        "TOKENIZERS_PARALLELISM": "false",
                        "OMP_NUM_THREADS": str(threads),
                        "OPENBLAS_NUM_THREADS": str(threads),
                    },
                )
                try:
                    code = process.wait(timeout=max(180, len(image_ids) * 120))
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
            if code:
                raise ValueError(
                    f"{stage} worker failed; inspect private log: {output / (stage + '.log')}"
                )
            result = json.loads((output / f"{stage}.json").read_text())
            result.pop("records", None)
            stages[stage] = {
                "wall_seconds": perf_counter() - start,
                "peak_mib": memory.peaks[stage] / 2**20,
                **result,
            }
        memory.phase = "retrieval"
        retrieval = evaluate(config)
        ocr_scores = evaluate_ocr(
            json.loads((output / "ocr.json").read_text())["records"], transcriptions
        )
        ocr_categories = ocr_by_category(
            json.loads((output / "ocr.json").read_text())["records"], transcriptions
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_kind": corpus_kind,
        "image_count": len(image_ids),
        "query_count": len(query_items),
        "precision": precision,
        "batch_size": batch_size,
        "threads": threads,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "ram_gib": psutil.virtual_memory().total / 2**30,
            "python": platform.python_version(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("realopen-lens", "numpy", "onnxruntime", "Pillow", "tokenizers", "psutil")
        },
        "artifacts": registry(),
        "preprocessing_version": 2,
        "corpus_sha256": {image_id: digest(data / image_id) for image_id in image_ids},
        "memory": {
            "peak_mib": max(memory.peaks.values()) / 2**20,
            "phase_peak_mib": {key: value / 2**20 for key, value in memory.peaks.items()},
            "sample_interval_ms": 20,
            "samples": memory.samples,
            "access_errors": memory.access_errors,
            "method": "Sum of sampled RSS of coordinator and recursive children",
        },
        "pipeline_seconds": perf_counter() - pipeline_start,
        "stages": stages,
        "retrieval": retrieval,
        "ocr_accuracy": ocr_scores,
        "ocr_accuracy_by_category": ocr_categories,
        "offline": {
            "local_artifacts_verified": True,
            "worker_python_network_blocked": True,
            "os_network_isolation_verified": False,
        },
        "limitations": [
            "No reference-framework parity check performed by this command",
            "First execution is process-cold, not guaranteed disk-cache-cold",
            "Peak memory is sampled; short spikes may be missed",
            "No 8 GB hardware claim follows from a run on a larger host",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    progress(f"Report: {output / 'report.md'}")
    return output / "report.json"
