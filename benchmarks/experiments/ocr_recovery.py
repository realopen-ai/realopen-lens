"""Resumable, isolated Tesseract regression probes against an explicit engine checkout."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
from time import perf_counter

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.metrics.quality import image_quality, quality_status
from benchmarks.metrics.statistics import distribution
from benchmarks.runtime.execution import stop_tree
from benchmarks.runtime.identity import source_hash
from benchmarks.runtime.resources import MemoryMonitor


def worker(args):
    from realopen_lens.runtime.offline import enforce
    from realopen_lens.ocr.tesseract import extract

    enforce()
    row = json.loads(sys.stdin.read())
    if digest(Path(row["path"])) != row["sha256"]:
        raise ValueError("OCR probe source changed")
    try:
        result = extract(Path(row["path"]), args.languages, args.threads, args.timeout)
        record = {"status": "ok", "actual": result}
    except Exception as error:
        record = {"status": "error", "error_type": type(error).__name__, "error": str(error)[:2000]}
    # Worker pipes must be portable even when Windows stdout defaults to CP1252.
    print(json.dumps(record))


def run(args):
    from realopen_lens.ocr import tesseract

    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    rows = inputs["images"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("OCR probe identities must be unique")
    root = Path(__file__).resolve().parents[2]
    executable = shutil.which("tesseract")
    if executable is None:
        raise ValueError("Tesseract is required for OCR probes")
    tesseract._environment.cache_clear()
    native_environment = tesseract._environment(executable, args.languages)
    identity = {
        "inputs_sha256": digest(args.inputs),
        "engine_sha256": source_hash(args.engine_root),
        "ocr_sha256": digest(args.engine_root / "realopen_lens/ocr/tesseract.py"),
        "harness_sha256": digest(Path(__file__)),
        "languages": args.languages,
        "threads": args.threads,
        "timeout": args.timeout,
        "tesseract": native_environment,
        "executable_sha256": digest(Path(executable)),
        "packages": {name: version(name) for name in ("numpy", "Pillow", "onnxruntime")},
        "host": {"system": platform.system(), "machine": platform.machine()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "config.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != identity:
        raise ValueError("OCR resume identity changed; choose a new output directory")
    save_json(config, identity)
    db = sqlite3.connect(args.output / "results.sqlite")
    try:
        db.execute("CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        done = {row[0] for row in db.execute("SELECT id FROM images")}
        for row in rows:
            if row["id"] in done:
                continue
            if source_hash(args.engine_root) != identity["engine_sha256"]:
                raise ValueError("OCR source changed during evaluation")
            started = perf_counter()
            record: dict
            memory = MemoryMonitor()
            env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(args.engine_root.resolve()), str(root))),
            }
            command = [
                sys.executable,
                "-m",
                "benchmarks.experiments.ocr_recovery",
                "worker",
                "--languages",
                args.languages,
                "--threads",
                str(args.threads),
                "--timeout",
                str(args.timeout),
            ]
            with memory:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    start_new_session=os.name != "nt",
                )
                try:
                    stdout, stderr = process.communicate(json.dumps(row), timeout=args.timeout + 60)
                    if process.returncode:
                        record = {
                            "status": "error",
                            "error_type": "WorkerExit",
                            "error": stderr[-2000:],
                            "returncode": process.returncode,
                        }
                    else:
                        record = json.loads(stdout)
                        record["worker_stderr"] = stderr[-2000:]
                        if record["status"] == "ok" and any(
                            record["actual"].get(key) != value
                            for key, value in native_environment.items()
                        ):
                            record.update(
                                status="error",
                                error_type="NativeEnvironmentChanged",
                                error="Tesseract or language packs changed during the probe",
                            )
                except subprocess.TimeoutExpired:
                    stop_tree(process)
                    record = {"status": "error", "error_type": "WorkerTimeout"}
                except BaseException:
                    if process.poll() is None:
                        stop_tree(process)
                    raise
            record.update(
                id=row["id"],
                input=row,
                seconds=perf_counter() - started,
                peak_tree_rss_mib=max(memory.peaks.values(), default=0) / 2**20,
            )
            if record["status"] == "ok":
                record["quality"] = image_quality(row, record["actual"]["text"])
                record["quality_status"] = quality_status(record["quality"])
            else:
                record["quality_status"] = "not_evaluated"
            with db:
                db.execute(
                    "INSERT INTO images VALUES (?,?)",
                    (row["id"], json.dumps(record, ensure_ascii=False)),
                )
            print(
                json.dumps(
                    {
                        "image_id": row["id"],
                        "status": record["status"],
                        "seconds": round(record["seconds"], 2),
                    }
                ),
                flush=True,
            )
        records = [
            json.loads(row[0]) for row in db.execute("SELECT payload FROM images ORDER BY id")
        ]
    finally:
        db.close()
    temporary = args.output / "images.jsonl.tmp"
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    temporary.replace(args.output / "images.jsonl")
    save_json(
        args.output / "summary.json",
        {
            "identity": identity,
            "expected_images": len(rows),
            "processed_images": len(records),
            "errors": sum(row["status"] != "ok" for row in records),
            "quality_evaluated_images": sum(
                row["quality_status"] != "not_evaluated" for row in records
            ),
            "latency_seconds": distribution([row["seconds"] for row in records]),
            "peak_tree_rss_mib": distribution([row["peak_tree_rss_mib"] for row in records]),
            "limitations": [
                "Small diagnostic control set, not full-corpus OCR accuracy.",
                "Execution recovery without a valid transcript does not prove OCR correctness.",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for mode in ("run", "worker"):
        command = subparsers.add_parser(mode)
        command.add_argument("--languages", default="eng+fra+ara")
        command.add_argument("--threads", type=int, choices=range(1, 9), default=2)
        command.add_argument("--timeout", type=float, default=30)
        if mode == "run":
            for name in ("inputs", "output", "engine-root"):
                command.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.timeout <= 120:
        parser.error("OCR timeout must be positive and at most 120 seconds")
    worker(args) if args.command == "worker" else run(args)


if __name__ == "__main__":
    main()
