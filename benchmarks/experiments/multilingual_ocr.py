"""Resumable serial language-set comparison using production OCR in isolated workers."""

import argparse
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
import subprocess
import sys
from time import perf_counter
import unicodedata

from benchmarks.datasets.multilingual import digest
from benchmarks.metrics.ocr import distance
from benchmarks.metrics.statistics import distribution
from benchmarks.runtime.execution import stop_tree
from benchmarks.runtime.resources import MemoryMonitor

PROFILES = ("baseline", "matched", "script", "all", "auto")
GROUPS = {
    "latin": "eng+fra+spa+por+deu+ita+ind+tur+vie",
    "arabic": "eng+ara+urd+fas",
    "devanagari": "eng+hin",
    "bengali": "eng+ben",
    "chinese": "eng+chi_sim+chi_tra+jpn+kor",
    "japanese": "eng+chi_sim+chi_tra+jpn+kor",
    "korean": "eng+chi_sim+chi_tra+jpn+kor",
}


def language_set(profile: str, row: dict, codes: list[str]) -> str:
    if profile == "auto":
        return "auto"
    if profile == "baseline":
        return "eng+fra+ara"
    if profile == "matched":
        return "+".join(dict.fromkeys([row["language"], "eng"]))
    if profile == "all":
        return "+".join(codes)
    return "rus+eng" if row["language"] == "rus" else GROUPS[row["script"]]


def normalized(text: str) -> str:
    # CER ignores layout whitespace and bidi format controls, but retains vowels/diacritics.
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text).casefold()
        if not char.isspace() and unicodedata.category(char) != "Cf"
    )


def summary(rows: list[dict]) -> dict:
    groups = {}
    for profile in sorted({r["profile"] for r in rows}):
        selected = [r for r in rows if r["profile"] == profile]

        def aggregate(values):
            return {
                "count": len(values),
                "errors": sum(r["status"] == "error" for r in values),
                "cer": sum(r["character_errors"] for r in values)
                / sum(r["reference_characters"] for r in values),
                "seconds": distribution([r["seconds"] for r in values]),
                "peak_tree_rss_mib": distribution([r["peak_tree_rss_mib"] for r in values]),
            }

        groups[profile] = {
            **aggregate(selected),
            "languages": {
                code: aggregate([r for r in selected if r["language"] == code])
                for code in sorted({r["language"] for r in selected})
            },
        }
    return groups


def run(dataset: Path, tessdata: Path, output: Path, profiles: list[str], threads: int) -> dict:
    from realopen_lens.ocr import tesseract

    if not 1 <= threads <= 8 or set(profiles) - set(PROFILES) or not profiles:
        raise ValueError("Use known profiles and 1–8 threads")
    tsv = tessdata / "configs/tsv"
    if not tsv.is_file() or "tessedit_create_tsv 1" not in tsv.read_text(encoding="utf-8"):
        raise ValueError("Missing TSV config; prepare benchmark assets with --fetch first")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "TESSDATA_PREFIX": str(tessdata.resolve())}
    executable = shutil.which("tesseract")
    if not executable:
        raise ValueError("Install Tesseract for this source benchmark")
    config = {
        "manifest_sha256": digest(dataset / "manifest.json"),
        "tsv_sha256": digest(tsv),
        "engine": subprocess.check_output(
            [executable, "--version"], text=True, env=environment
        ).splitlines(),
        "engine_sha256": digest(Path(executable)),
        "ocr_sha256": digest(Path(tesseract.__file__)),
        "runner_sha256": digest(Path(__file__)),
        "threads": threads,
        "profiles": profiles,
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {name: version(name) for name in ("Pillow", "numpy", "onnxruntime", "psutil")},
        "packs": {p.stem: digest(p) for p in sorted(tessdata.glob("*.traineddata"))},
    }
    if "auto" in profiles:
        from realopen_lens.ocr.automatic import policy

        config["automatic_policy"] = policy()
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Run configuration changed; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    codes = list(dict.fromkeys(r["language"] for r in manifest["images"]))
    database = sqlite3.connect(output / "results.sqlite")
    database.execute(
        "CREATE TABLE IF NOT EXISTS results "
        "(id TEXT, profile TEXT, payload TEXT, PRIMARY KEY(id,profile))"
    )
    done = set(database.execute("SELECT id,profile FROM results"))
    for i, row in enumerate(manifest["images"]):
        path = dataset / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Changed image {row['id']}")
        order = profiles.copy()
        random.Random(20261026 + i).shuffle(order)
        for profile in order:
            if profile == "auto" and policy() != config["automatic_policy"]:
                raise ValueError("Automatic routing policy changed; use a new output")
            if digest(Path(tesseract.__file__)) != config["ocr_sha256"]:
                raise ValueError(
                    "OCR source changed during run; preserve results and use a new output"
                )
            if (row["id"], profile) in done:
                continue
            languages = language_set(profile, row, codes)
            start = perf_counter()
            with MemoryMonitor() as memory:
                process = subprocess.Popen(
                    [sys.executable, "-m", "benchmarks.experiments.multilingual_ocr", "--worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                    start_new_session=sys.platform != "win32",
                )
                try:
                    stdout, stderr = process.communicate(
                        json.dumps(
                            {
                                "path": str(path.resolve()),
                                "languages": languages,
                                "threads": threads,
                            }
                        ).encode(),
                        timeout=50,
                    )
                    if process.returncode:
                        raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1500:])
                    result = json.loads(stdout)
                except Exception as error:
                    if process.poll() is None:
                        stop_tree(process)
                    result = {
                        "status": "error",
                        "text": "",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
            expected, actual = normalized(row["text"]), normalized(result.get("text", ""))
            errors = distance(expected, actual)
            record = {
                **row,
                "profile": profile,
                "languages": languages,
                "actual": result,
                "status": result.get("status", "ok"),
                "seconds": perf_counter() - start,
                "peak_tree_rss_mib": max(memory.peaks.values(), default=0) / 2**20,
                "memory_access_errors": memory.access_errors,
                "memory_samples": memory.samples,
                "character_errors": errors,
                "reference_characters": len(expected),
                "cer": errors / len(expected),
                "quality_status": "degraded" if errors / len(expected) > 0.2 else "pass",
            }
            with database:
                database.execute(
                    "INSERT INTO results VALUES (?,?,?)",
                    (row["id"], profile, json.dumps(record, ensure_ascii=False)),
                )
        print(f"OCR {i + 1}/{len(manifest['images'])}: {row['id']}", flush=True)
    rows = [
        json.loads(r[0])
        for r in database.execute("SELECT payload FROM results ORDER BY id,profile")
    ]
    database.close()
    (output / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    result = summary(rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--tessdata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES[:-1]))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        from realopen_lens.runtime.offline import enforce
        from realopen_lens.ocr.tesseract import extract

        enforce()
        request = json.load(sys.stdin)
        try:
            result = extract(Path(request["path"]), request["languages"], request["threads"])
        except Exception as error:
            result = {
                "status": "error",
                "text": "",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        print(json.dumps(result, ensure_ascii=False))
    else:
        if not all((args.dataset, args.tessdata, args.output)):
            parser.error("--dataset, --tessdata and --output are required")
        run(args.dataset, args.tessdata, args.output, args.profiles.split(","), args.threads)


if __name__ == "__main__":
    main()
