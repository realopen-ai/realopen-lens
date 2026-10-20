"""Isolated, local encoder probes with per-input outputs and reference comparisons."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks.datasets.photos import save_json
from benchmarks.metrics.statistics import distribution
from benchmarks.runtime.resources import MemoryMonitor
from benchmarks.runtime.execution import stop_tree
from realopen_lens.models.manager import artifact_path, digest
from realopen_lens.runtime.offline import enforce

STAGES = ("vision", "siglip2", "e5-query", "e5-passage")


def save_item(path: Path, record: dict, vector: np.ndarray | None) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            record=np.asarray(json.dumps(record, ensure_ascii=False)),
            vector=np.asarray(vector, dtype=np.float32) if vector is not None else np.empty(0),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def worker(args) -> None:
    enforce()
    config = json.loads(args.inputs.read_text(encoding="utf-8"))
    kind = "e5" if args.stage.startswith("e5") else args.stage
    rows = config[
        "images" if kind == "vision" else "passages" if args.stage == "e5-passage" else "queries"
    ]
    identity = {
        "inputs_sha256": digest(args.inputs),
        "backend": args.backend,
        "stage": args.stage,
        "threads": args.threads,
        "siglip_lowercase": args.siglip_lowercase,
        "python": platform.python_version(),
        "host": {"system": platform.system(), "machine": platform.machine()},
        "packages": {},
        "artifacts": {},
        "probe_sha256": digest(Path(__file__)),
        "reference_adapter_sha256": digest(
            Path(__file__).parents[1] / "runtime/encoder_reference.py"
        ),
        "inference_sha256": digest(
            Path(__file__).resolve().parents[2] / "src/realopen_lens/encoders/onnx.py"
        ),
    }
    for name in ("numpy", "Pillow", "onnxruntime", "tokenizers", "torch", "transformers"):
        try:
            identity["packages"][name] = version(name)
        except PackageNotFoundError:
            identity["packages"][name] = None
    if args.backend == "reference":
        model_name = "siglip2" if kind in ("vision", "siglip2") else "e5"
        lock = json.loads(args.reference_lock.read_text(encoding="utf-8"))["models"][model_name]
        directory = args.reference_models / model_name / lock["revision"]
        for filename, spec in lock["files"].items():
            path = directory / filename
            if path.stat().st_size != spec["size"] or digest(path) != spec["sha256"]:
                raise ValueError("Reference artifact failed its frozen checksum: " + filename)
        identity["artifacts"] = lock
    else:
        model_name = "siglip2" if kind in ("vision", "siglip2") else "e5"
        key = ("vision_" if kind == "vision" else "text_") + args.backend
        graph = artifact_path(args.models, model_name, key)
        tokenizer = artifact_path(args.models, model_name, "tokenizer")
        from realopen_lens.models.manager import registry

        specs = registry()["models"][model_name]["files"]
        for path, name in ((graph, key), (tokenizer, "tokenizer")):
            if path.stat().st_size != specs[name]["size"] or digest(path) != specs[name]["sha256"]:
                raise ValueError("ONNX artifact failed its frozen checksum")
            identity["artifacts"][name] = specs[name]
    if (args.output / "report.json").exists():
        previous = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        if previous["identity"] == identity and previous["errors"] == 0:
            if (
                digest(args.output / "vectors.npy") != previous["vectors_sha256"]
                or digest(args.output / "records.jsonl") != previous["records_sha256"]
            ):
                raise ValueError("Completed probe outputs changed")
            print("Reusing completed encoder probe:", json.dumps(str(args.output)))
            return
        raise ValueError("Preserve failed/different encoder runs; choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Interrupted probe identity changed; choose a new output directory")
    else:
        temporary_config = config_path.with_suffix(".json.tmp")
        save_json(temporary_config, identity)
        temporary_config.replace(config_path)
    items = args.output / "items"
    items.mkdir(exist_ok=True)
    memory = MemoryMonitor()
    memory.phase = "encoder_probe"
    records: list[dict] = []
    vectors: list[np.ndarray] = []
    encoder: Any
    started = perf_counter()
    with memory:
        if args.backend == "reference":
            from benchmarks.runtime.encoder_reference import ReferenceEncoder

            encoder = ReferenceEncoder(directory, kind, args.threads, args.siglip_lowercase)
        else:
            from realopen_lens.encoders.onnx import TextEncoder, VisionEncoder

            encoder = (
                VisionEncoder(graph, args.threads)
                if kind == "vision"
                else TextEncoder(graph, tokenizer, kind, args.threads)
            )
        load_seconds = perf_counter() - started
        with (args.output / "records.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                checkpoint = items / (hashlib.sha256(row["id"].encode()).hexdigest() + ".npz")
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as data:
                        record = json.loads(str(data["record"].item()))
                        saved_vector = data["vector"]
                    if record["id"] != row["id"]:
                        raise ValueError("Encoder checkpoint identity differs")
                    if record["status"] == "ok":
                        if (
                            saved_vector.shape != (768 if kind in ("vision", "siglip2") else 384,)
                            or not np.isfinite(saved_vector).all()
                        ):
                            raise ValueError("Invalid encoder checkpoint vector")
                        record["vector_row"] = len(vectors)
                        vectors.append(saved_vector)
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    continue
                start = perf_counter()
                saved_vector = None
                record = {
                    "id": row["id"],
                    "language": row.get("language"),
                    "category": row.get("category"),
                }
                try:
                    if kind == "vision":
                        if digest(Path(row["path"])) != row["sha256"]:
                            raise ValueError("Frozen probe image changed")
                        vector = encoder.encode([Path(row["path"])])
                    else:
                        vector = encoder.encode([row["text"]], passage=args.stage == "e5-passage")
                    if (
                        vector.shape != (1, 768 if kind in ("vision", "siglip2") else 384)
                        or not np.isfinite(vector).all()
                    ):
                        raise ValueError("Invalid encoder output shape or values")
                    if kind != "vision":
                        if args.backend == "reference":
                            ids = encoder.last_inputs["input_ids"][0].tolist()
                        else:
                            text = (
                                row["text"].lower()
                                if kind == "siglip2"
                                else (
                                    ("passage: " if args.stage == "e5-passage" else "query: ")
                                    + row["text"]
                                )
                            )
                            ids = encoder.tokenizer.encode(text).ids
                        record["input_ids"] = ids
                    elif args.backend == "reference":
                        # Compare official resize/normalization against the production adapter.
                        from PIL import Image

                        with Image.open(row["path"]) as image:
                            resized = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
                            pixels = (
                                np.asarray(resized, dtype=np.float32) * np.float32(1 / 255) - 0.5
                            ) / 0.5
                            pixels = pixels.transpose(2, 0, 1)[None]
                        record["pixel_max_abs_difference"] = float(
                            np.max(np.abs(pixels - encoder.last_inputs["pixel_values"]))
                        )
                    record.update(status="ok", vector_row=len(vectors))
                    saved_vector = vector[0]
                    vectors.append(saved_vector)
                except Exception as error:
                    record.update(
                        status="error", error_type=type(error).__name__, error=str(error)[:1000]
                    )
                record["seconds"] = perf_counter() - start
                save_item(checkpoint, record, saved_vector)
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
    np.save(args.output / "vectors.npy", np.asarray(vectors, dtype=np.float32))
    errors = sum(row["status"] != "ok" for row in records)
    report_path = args.output / "report.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    save_json(
        temporary_report,
        {
            "identity": identity,
            "inputs": len(rows),
            "errors": errors,
            "load_seconds": load_seconds,
            "inference_seconds": distribution([row["seconds"] for row in records]),
            "peak_tree_rss_mib": max(memory.peaks.values(), default=0) / 2**20,
            "memory_samples": memory.samples,
            "memory_access_errors": memory.access_errors,
            "vectors_sha256": digest(args.output / "vectors.npy"),
            "records_sha256": digest(args.output / "records.jsonl"),
            "checkpointed_inputs": len(records),
            "memory_scope": "Last invocation's worker process tree; "
            "per-input timings survive resume",
        },
    )
    temporary_report.replace(report_path)
    if errors:
        raise ValueError(f"{errors} encoder inputs failed; retained in {args.output}")
    print(f"{args.backend}/{args.stage}: {len(rows)} inputs, {errors} errors", flush=True)


def compare(left: Path, right: Path, output: Path) -> dict:
    configs = [
        json.loads((path / "report.json").read_text(encoding="utf-8")) for path in (left, right)
    ]
    if any(config["errors"] for config in configs) or any(
        configs[0]["identity"][key] != configs[1]["identity"][key]
        for key in ("inputs_sha256", "stage")
    ):
        raise ValueError("Encoder parity requires complete, matching probe inputs and stages")
    for path, config in zip((left, right), configs, strict=True):
        if (
            digest(path / "vectors.npy") != config["vectors_sha256"]
            or digest(path / "records.jsonl") != config["records_sha256"]
        ):
            raise ValueError("Encoder probe outputs changed")
    arrays = [np.load(path / "vectors.npy", allow_pickle=False) for path in (left, right)]
    records = [
        list(map(json.loads, (path / "records.jsonl").read_text(encoding="utf-8").splitlines()))
        for path in (left, right)
    ]
    if (
        [r["id"] for r in records[0]] != [r["id"] for r in records[1]]
        or arrays[0].shape != arrays[1].shape
        or len(arrays[0]) != len(records[0])
    ):
        raise ValueError("Encoder outputs have different identities/shapes")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Nonfinite parity output")
    individual = []
    groups: dict[str, list] = {}
    for index, (a, b) in enumerate(zip(*records, strict=True)):
        row = {
            "id": a["id"],
            "cosine": float(arrays[0][index] @ arrays[1][index]),
            "max_abs_difference": float(np.max(np.abs(arrays[0][index] - arrays[1][index]))),
            "input_ids_equal": a.get("input_ids") == b.get("input_ids"),
            "language": a.get("language"),
            "category": a.get("category"),
        }
        individual.append(row)
        for group in (
            "all",
            "language:" + str(row["language"]),
            "category:" + str(row["category"]),
        ):
            groups.setdefault(group, []).append(row)
    report = {
        "version": 1,
        "left": configs[0],
        "right": configs[1],
        "individual": individual,
        "groups": {
            key: {
                "cosine": distribution([r["cosine"] for r in rows]),
                "token_mismatches": sum(not r["input_ids_equal"] for r in rows),
            }
            for key, rows in groups.items()
        },
        "limitations": [
            "Embedding agreement is not a retrieval accuracy guarantee.",
            "Different official revisions/exporters may contribute to disagreements.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--inputs", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--backend", choices=("int8", "fp32", "reference"), required=True)
    probe.add_argument("--stage", choices=STAGES, required=True)
    probe.add_argument("--models", type=Path, default=Path("models"))
    probe.add_argument("--reference-models", type=Path, default=Path("models/encoder-reference"))
    probe.add_argument(
        "--reference-lock",
        type=Path,
        default=Path("benchmarks/sources/encoder-reference.lock.json"),
    )
    probe.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    probe.add_argument(
        "--stage-timeout",
        type=float,
        default=3600,
        help="Bound the isolated encoder worker's wall time in seconds",
    )
    probe.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    probe.add_argument(
        "--siglip-lowercase",
        action="store_true",
        help="Match production lowercase inputs for a separate reference graph comparison",
    )
    comparison = commands.add_parser("compare")
    for key in ("left", "right", "output"):
        comparison.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        if args.siglip_lowercase and (args.backend != "reference" or args.stage != "siglip2"):
            parser.error("--siglip-lowercase applies only to reference SigLIP text probes")
        if not np.isfinite(args.stage_timeout) or not 0 < args.stage_timeout <= 86400:
            parser.error("Stage timeout must be positive and at most 86400 seconds")
        if args.internal_worker:
            worker(args)
        else:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.experiments.encoder_parity",
                    *sys.argv[1:],
                    "--internal-worker",
                ],
                start_new_session=os.name != "nt",
            )
            try:
                code = process.wait(timeout=args.stage_timeout)
            except BaseException:
                if process.poll() is None:
                    stop_tree(process)
                raise
            if code:
                raise SystemExit(code)
    else:
        compare(args.left, args.right, args.output)


if __name__ == "__main__":
    main()
