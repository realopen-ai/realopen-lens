"""Index every language through production workers, then query the saved library."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter

from benchmarks.datasets.multilingual import digest
from benchmarks.metrics.retrieval import metrics
from benchmarks.runtime.resources import MemoryMonitor
from realopen_lens.indexing.service import run_index
from realopen_lens.retrieval.library import LibrarySession


def run(dataset: Path, tessdata: Path, models: Path, output: Path, precision: str):
    from realopen_lens.ocr import tesseract
    from realopen_lens.models.manager import registry

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    config = {
        "manifest_sha256": digest(dataset / "manifest.json"),
        "model_registry": registry(),
        "siglip_precision": precision,
        "ocr_sha256": digest(Path(tesseract.__file__)),
        "runner_sha256": digest(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError("Configuration changed; use a new output directory")
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.environ["TESSDATA_PREFIX"] = str(tessdata.resolve())
    tesseract._environment.cache_clear()
    images = manifest["images"]
    for row in images:
        if digest(dataset / row["path"]) != row["sha256"]:
            raise ValueError(f"Changed image: {row['id']}")
    codes = list(dict.fromkeys(row["language"] for row in images))
    indexes = []
    with MemoryMonitor() as memory:
        for i, code in enumerate(codes):
            memory.phase = f"index-{code}"
            indexed = run_index(
                {
                    "root": str((dataset / code).resolve()),
                    "store": str((output / "index").resolve()),
                    "collection": code,
                    "models": str(models.resolve()),
                    "threads": 2,
                    "languages": "+".join(dict.fromkeys([code, "eng"])),
                    "rehash": False,
                    "siglip_precision": precision,
                }
            )
            indexes.append(asdict(indexed))
            if indexed.failed_files:
                raise ValueError(f"Indexing failed for {code}")
            (output / "indexes.json").write_text(
                json.dumps(indexes, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Indexed {code} {i + 1}/{len(codes)}", flush=True)
        lookup = {row["sha256"]: row for row in images}
        query_file = output / "queries.jsonl"
        queries = (
            [json.loads(line) for line in query_file.read_text(encoding="utf-8").splitlines()]
            if query_file.exists()
            else []
        )
        done = {(row["id"], row["mode"]) for row in queries}
        session = LibrarySession(index_dir=output / "index", models=models, threads=2)
        try:
            for row in images:
                for mode in ("lexical", "semantic", "hybrid"):
                    if (row["id"], mode) in done:
                        continue
                    memory.phase = row["id"] + "-" + mode
                    start = perf_counter()
                    try:
                        result = session.search(
                            row["query"],
                            mode=mode,
                            limit=100,
                            use_feedback=False,
                            interpret_metadata=False,
                        )
                        ranked = [record["image_id"] for record in result["results"]]
                        relevant = [row["sha256"]]
                        if mode != "lexical":
                            ranked = [
                                key for key in ranked if lookup[key]["language"] != row["language"]
                            ]
                            relevant = [
                                r["sha256"]
                                for r in images
                                if r["language"] != row["language"] and r["topic"] == row["topic"]
                            ]
                        record = {
                            "id": row["id"],
                            "language": row["language"],
                            "mode": mode,
                            "query": row["query"],
                            "status": "ok",
                            "result": result,
                            "metrics": metrics(ranked, relevant),
                        }
                    except Exception as error:
                        record = {
                            "id": row["id"],
                            "language": row["language"],
                            "mode": mode,
                            "query": row["query"],
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    record["seconds"] = perf_counter() - start
                    record["peak_tree_rss_mib"] = memory.peaks[memory.phase] / 2**20
                    queries.append(record)
                    with (output / "queries.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Queried {row['id']}", flush=True)
        finally:
            session.close()
        (output / "resources.json").write_text(
            json.dumps(
                {
                    "phase_peak_mib": {k: v / 2**20 for k, v in memory.peaks.items()},
                    "access_errors": memory.access_errors,
                    "sample_interval_ms": 20,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "tessdata", "models", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--siglip-precision", choices=("int8", "fp32"), default="int8")
    args = parser.parse_args()
    run(args.dataset, args.tessdata, args.models, args.output, args.siglip_precision)


if __name__ == "__main__":
    main()
