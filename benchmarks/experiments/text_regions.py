"""Experimental Tesseract text-localization comparison, with resumable per-case results."""

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import monotonic, perf_counter
from importlib.metadata import version
from benchmarks.runtime.execution import run_stage
from benchmarks.datasets.references import inputs
from benchmarks.metrics.records import record
from benchmarks.runtime.resources import MemoryMonitor
from benchmarks.storage.results import connect, save_stage
from benchmarks.runtime.identity import source_hash
from benchmarks.metrics.statistics import summarize
from realopen_lens.models.manager import digest

VARIANTS = ("baseline", "gate", "mask")


def worker(path: Path, variant: str):
    from realopen_lens.runtime.offline import enforce

    enforce()
    from realopen_lens.ocr.tesseract import extract
    from realopen_lens.ocr.regions import TextRegionDetector

    config = json.loads(path.read_text())
    db = connect(path.parent / "results.sqlite")
    done = {r[0] for r in db.execute("select image_id from stages where stage=?", (variant,))}
    pending = [r for r in config["images"] if r["id"] not in done]
    if not pending:
        return
    detector = None
    with MemoryMonitor() as memory:
        started = perf_counter()
        if variant != "baseline":
            detector = TextRegionDetector(Path(config["model"]), config["threads"])
        with db:
            db.execute(
                "insert or replace into metadata values (?,?)",
                (
                    variant + "_environment",
                    json.dumps(
                        {
                            "load_seconds": perf_counter() - started,
                            "detector": detector.info if detector else None,
                        }
                    ),
                ),
            )
        for n, row in enumerate(pending):
            memory.phase = row["id"]
            memory._sample()
            started = perf_counter()
            with db:
                db.execute(
                    "insert or replace into metadata values (?,?)",
                    (
                        variant + "_current",
                        json.dumps({"id": row["id"], "started_at": monotonic()}),
                    ),
                )
            try:
                result = {
                    **extract(
                        Path(row["input_path"]),
                        threads=config["threads"],
                        detector=detector,
                        region_mode=variant if detector else "mask",
                    ),
                    "status": "ok",
                }
            except Exception as error:
                result = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "warnings": [],
                }
            memory._sample()
            result.update(
                seconds=perf_counter() - started, peak_tree_rss_mib=memory.peaks[row["id"]] / 2**20
            )
            save_stage(db, row["id"], variant, result)
            if (n + 1) % 25 == 0:
                print(f"{variant} {n+1}/{len(pending)}", flush=True)
        memory.phase = "shutdown"
    with db:
        db.execute(
            "insert or replace into metadata values (?,?)",
            (
                variant + "_resources",
                json.dumps(
                    {
                        "peak_tree_rss_mib": max(memory.peaks.values()) / 2**20,
                        "samples": memory.samples,
                        "access_errors": memory.access_errors,
                    }
                ),
            ),
        )
    db.close()


def report(root: Path):
    config = json.loads((root / "config.json").read_text())
    db = connect(root / "results.sqlite")
    report = {"config": config, "completed_at": datetime.now(UTC).isoformat(), "variants": {}}
    for variant in config["variants"]:
        actual = {
            k: json.loads(v)
            for k, v in db.execute("select image_id,payload from stages where stage=?", (variant,))
        }
        if set(actual) != {r["id"] for r in config["images"]}:
            raise ValueError("Incomplete run; resume first")
        rows = []
        for row in config["images"]:
            r = record(row, actual[row["id"]])
            if row.get("negative_text"):
                r["quality_status"] = (
                    "pass"
                    if r["actual"]["status"] == "ok" and not r["actual"]["text"].strip()
                    else "fail"
                )
            rows.append(r)
        (root / (variant + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        scored = [
            r
            for r in rows
            if r["expected"]["decode"] == "success" and (r.get("truth") or r.get("key_phrases"))
        ]
        negatives = [r for r in rows if r.get("negative_text")]
        report["variants"][variant] = {
            "aggregate": summarize(rows),
            "positive_scored": len(scored),
            "positive_passed": sum(r["quality_status"] == "pass" for r in scored),
            "negative_scored": len(negatives),
            "negative_empty": sum(r["quality_status"] == "pass" for r in negatives),
            "metadata": {
                k: json.loads(v)
                for k, v in db.execute(
                    "select key,value from metadata where key like ?", (variant + "_%",)
                )
            },
        }
    (root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for variant, r in report["variants"].items():
        print(
            variant,
            json.dumps({k: v for k, v in r.items() if k not in ["aggregate", "metadata"]}),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("data/benchmark-500-v2"))
    p.add_argument("--controls", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model", type=Path, default=Path("models/text-detector/PP-OCRv6_det_small.onnx")
    )
    p.add_argument("--threads", type=int, choices=range(1, 33), default=2)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--worker", choices=VARIANTS)
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    if a.worker:
        return worker(a.config, a.worker)
    rows = inputs(a.dataset)[: a.limit]
    for r in rows:
        r["input_path"] = str((a.dataset / r["path"]).resolve())
        r["corpus"] = "main"
    if a.controls:
        controls = json.loads((a.controls / "manifest.json").read_text())["images"]
        for r in controls:
            if digest(a.controls / r["path"]) != r["sha256"]:
                raise ValueError("Control image changed")
            r["input_path"] = str((a.controls / r["path"]).resolve())
            r["corpus"] = "controls"
        rows += controls
    config = {
        "images": rows,
        "model": str(a.model.resolve()),
        "model_sha256": digest(a.model),
        "threads": a.threads,
        "variants": a.variants,
        "source": {"engine": source_hash(Path("src")), "harness": source_hash(Path("benchmarks"))},
        "packages": {
            n: version(n) for n in ["Pillow", "numpy", "onnxruntime", "opencv-python-headless"]
        },
        "host": {"platform": platform.platform(), "python": sys.version},
        "languages": "eng+fra+ara",
    }
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Run identity changed; use a new output")
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    if not (a.output / "engine-source").exists():
        for source, target in [("src", "engine-source"), ("benchmarks", "harness-source")]:
            shutil.copytree(source, a.output / target, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ["pyproject.toml", "uv.lock"]:
            shutil.copyfile(name, a.output / name)
    db = connect(a.output / "results.sqlite")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path("src").resolve()) + os.pathsep + str(Path.cwd()),
        "OMP_NUM_THREADS": str(a.threads),
        "OPENBLAS_NUM_THREADS": str(a.threads),
        "VECLIB_MAXIMUM_THREADS": str(a.threads),
    }
    for variant in a.variants:
        print("Running Tesseract", variant, flush=True)
        events = run_stage(
            [
                sys.executable,
                "-m",
                "benchmarks.experiments.text_regions",
                "--output",
                str(a.output),
                "--worker",
                variant,
                "--config",
                str(path),
            ],
            db,
            variant,
            env,
            a.output / (variant + ".log"),
            lambda key, data: save_stage(db, key, variant, data),
            image_timeout=60,
            stage_timeout=7200,
        )
        with (a.output / "events.jsonl").open("a") as stream:
            for e in events:
                stream.write(json.dumps(e) + "\n")
    db.close()
    report(a.output)


if __name__ == "__main__":
    main()
