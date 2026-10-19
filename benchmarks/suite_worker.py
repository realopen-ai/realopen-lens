"""Bounded stage workers; commit each image independently so failures are resumable."""

import importlib.util
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter, monotonic


from benchmarks.resources import MemoryMonitor
from benchmarks.suite import connect, save_stage
from realopen_lens.inference import TextEncoder, VisionEncoder
from realopen_lens.models import artifact_path
from realopen_lens.offline import enforce


def baseline_ocr(config: dict, row: dict) -> dict:
    path = Path(config["engine_root"]).parent / "benchmarks/worker.py"
    spec = importlib.util.spec_from_file_location("preserved_baseline_worker", path)
    if spec is None or spec.loader is None:
        raise ValueError("Baseline worker is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.ocr(
        {
            "data": config["dataset"],
            "images": [row["path"]],
            "tesseract": shutil.which("tesseract"),
            "threads": config["threads"],
            "languages": config["languages"],
        }
    )
    return {
        "text": result["records"][0]["text"],
        "version": result["version"],
        "language_pack_sha256": result["language_pack_sha256"],
        "warnings": [],
        "words": None,
        "confidence": None,
        "attempts": [{"mode": "original-psm3"}],
    }


def main() -> None:
    enforce()
    config_path, stage = sys.argv[1:]
    config = json.loads(Path(config_path).read_text())
    output = Path(config_path).parent
    db = connect(output / "results.sqlite")
    done = {row[0] for row in db.execute("SELECT image_id FROM stages WHERE stage=?", (stage,))}
    items = config["queries"] if stage.endswith("_queries") else config["images"]
    pending = [row for row in items if row["id"] not in done]
    if not pending:
        db.close()
        return
    root = Path(config["models"])
    threads = config["threads"]
    encoder: VisionEncoder | TextEncoder | None = None
    start = perf_counter()
    if stage == "vision":
        encoder = VisionEncoder(artifact_path(root, "siglip2", "vision_int8"), threads)
    if stage in ("semantic", "visual_queries", "semantic_queries"):
        kind = "siglip2" if stage == "visual_queries" else "e5"
        encoder = TextEncoder(
            artifact_path(root, kind, "text_int8"),
            artifact_path(root, kind, "tokenizer"),
            kind,
            threads,
        )
    with db:
        db.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?,?)",
            (
                stage + "_environment",
                json.dumps(
                    {
                        "load_seconds": perf_counter() - start,
                        "worker_python": sys.version,
                        "stage": stage,
                    }
                ),
            ),
        )
    with MemoryMonitor() as monitor:
        for number, row in enumerate(pending):
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                    (stage + "_current", json.dumps({"id": row["id"], "started_at": monotonic()})),
                )
            monitor.phase = row["id"]
            start = perf_counter()
            vector = None
            try:
                warnings = []
                if stage == "ocr":
                    if config["variant"] == "baseline":
                        data = baseline_ocr(config, row)
                    else:
                        from realopen_lens.ocr import extract

                        data = extract(
                            Path(config["dataset"]) / row["path"], config["languages"], threads
                        )
                elif stage == "vision":
                    assert isinstance(encoder, VisionEncoder)
                    vector = encoder.encode([Path(config["dataset"]) / row["path"]])
                    if row["frames"] > 1:
                        warnings.append("Only first frame/page processed")
                    data = {"embedding_shape": list(vector.shape), "warnings": warnings}
                elif stage == "semantic":
                    assert isinstance(encoder, TextEncoder)
                    found = db.execute(
                        "SELECT payload FROM stages WHERE image_id=? AND stage='ocr'", (row["id"],)
                    ).fetchone()
                    ocr = json.loads(found[0]) if found else {}
                    if ocr.get("status") != "ok":
                        data = {
                            "status": "blocked",
                            "warnings": ["OCR failed; semantic stage cannot run"],
                            "passages": 0,
                        }
                    else:
                        text = ocr["text"]
                        truncated = len(text) > 32000
                        passages = encoder.passages(text[:32000]) if text.strip() else []
                        if len(passages) > 16:
                            truncated = True
                            passages = passages[:16]
                        if passages:
                            vector = encoder.encode(passages, passage=True)
                        data = {
                            "passages": len(passages),
                            "warnings": (
                                ["OCR passage budget exceeded; first 16 / 32000 characters only"]
                                if truncated
                                else []
                            ),
                        }
                else:
                    assert isinstance(encoder, TextEncoder)
                    vector = encoder.encode([row["text"]])
                    data = {"warnings": []}
                data.setdefault("status", "ok")
            except Exception as error:
                data = {
                    "status": (
                        "timeout" if isinstance(error, subprocess.TimeoutExpired) else "error"
                    ),
                    "error_type": type(error).__name__,
                    "error": str(error)[:1500],
                    "traceback": traceback.format_exc(limit=5)[-3000:],
                    "warnings": [],
                }
            monitor._sample()
            data.update(
                seconds=perf_counter() - start,
                peak_tree_rss_mib=monitor.peaks[row["id"]] / 2**20,
                memory_access_errors=monitor.access_errors,
            )
            save_stage(db, row["id"], stage, data, vector)
            if number % 25 == 0:
                print(
                    f'{stage}: {number+1}/{len(pending)}; {row["id"]}; {data["status"]}', flush=True
                )
    db.close()


if __name__ == "__main__":
    main()
