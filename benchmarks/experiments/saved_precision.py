"""Prepare a controlled precision baseline from a verified saved collection."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.runtime.collections import snapshot
from benchmarks.runtime.resources import MemoryMonitor, write_resources
from realopen_lens.indexing import index_collection
from realopen_lens.indexing.service import pipeline_config
from realopen_lens.models.manager import digest
from realopen_lens.runtime.jobs import writer_lock
from realopen_lens.runtime.offline import enforce
from realopen_lens.storage.artifacts import atomic_bytes, load_manifest, matrices, valid_analysis
from realopen_lens.storage.catalog import Catalog, local_path


def prepare(args) -> dict:
    corpus, _ = load(args.corpus, args.judgments)
    state = snapshot(
        args.index_dir,
        args.collection,
        corpus,
        corpus_prefix=args.corpus_prefix,
        allow_extra_images=args.allow_extra_images,
    )
    target = (args.output / "int8-index").resolve()
    source_root = args.index_dir.resolve()
    if target.is_relative_to(source_root) or source_root.is_relative_to(target):
        raise ValueError("Comparison index must be separate from the source index")
    if target.is_relative_to(Path(state["root"]).resolve()):
        raise ValueError("Comparison output must be outside the image root")
    source = Catalog(args.index_dir, readonly=True)
    try:
        _, generation = source.snapshot(args.collection)
        if generation["id"] != state["generation"]:
            raise ValueError("Source generation changed during preparation")
        manifest = load_manifest(source, generation)
        original = manifest["config"]
        if original["spec"].get("siglip_precision", "int8") != "fp32":
            raise ValueError("This controlled baseline requires an FP32 source collection")
        options = {key: original[key] for key in ("root", "models", "threads", "languages")} | {
            "store": str(target),
            "collection": args.collection,
            "siglip_precision": "int8",
            "rehash": False,
            "log": True,
        }
        config = pipeline_config(options)
        expected = dict(original["spec"], siglip_precision="int8")
        if config["spec"] != expected:
            raise ValueError(
                "Source processing/runtime changed; cannot reuse fixed OCR/E5 controls"
            )
        protocol = {
            "source": state,
            "source_index": str(source_root),
            "target": str(target),
            "target_pipeline": config["pipeline"],
            "corpus_sha256": digest(args.corpus),
            "judgments_sha256": digest(args.judgments),
            "method": (
                "Recompute INT8 vision through production indexing; "
                "retain verified metadata/OCR/E5"
            ),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        identity = args.output / "preparation.json"
        if target.exists() and not identity.exists():
            raise ValueError(
                "Refusing to seed an existing index without matching preparation provenance"
            )
        if identity.exists() and json.loads(identity.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Preparation identity changed; use a new output directory")
        save_json(identity, protocol)
        failures = [r for r in source.files(generation["job_id"]) if r["status"] == "error"]
        save_json(args.output / "source-input-failures.json", {"files": failures})
        with writer_lock(target):
            destination = Catalog(target)
            try:
                collection = destination.ensure_collection(args.collection, Path(options["root"]))
                job = destination.create_job(collection["id"], config)
                try:
                    for row in manifest["images"]:
                        for stage in ("metadata", "ocr", "semantic"):
                            record = source.analysis(original["pipeline"], row["image_id"], stage)
                            if not valid_analysis(source, record):
                                raise ValueError(
                                    f"Fixed-control preparation requires valid cached {stage}"
                                )
                            assert record is not None
                            if record["artifact"]:
                                path = local_path(target, record["artifact"])
                                if not path.is_file() or digest(path) != record["checksum"]:
                                    atomic_bytes(
                                        path,
                                        local_path(source.root, record["artifact"]).read_bytes(),
                                    )
                                if digest(path) != record["checksum"]:
                                    raise ValueError("Copied control artifact checksum differs")
                            destination.save_analysis(
                                config["pipeline"],
                                row["image_id"],
                                stage,
                                json.loads(record["payload"]),
                                record["artifact"],
                                record["checksum"],
                                job_id=job,
                            )
                    destination.finish(job, "completed", result={"purpose": protocol["method"]})
                except BaseException as error:
                    destination.finish(job, "failed", error=str(error))
                    raise
            finally:
                destination.close()
    finally:
        source.close()
    memory = MemoryMonitor()
    memory.phase = "production_int8_vision_index"
    with memory:
        result = index_collection(
            Path(options["root"]),
            collection=args.collection,
            index_dir=target,
            models=Path(options["models"]),
            threads=options["threads"],
            languages=options["languages"],
            siglip_precision="int8",
            log=True,
        )
    write_resources(args.output, memory)
    number = len(list(args.output.glob("index-attempt-*.json"))) + 1
    save_json(args.output / f"index-attempt-{number:03}.json", asdict(result))
    after = snapshot(
        args.index_dir,
        args.collection,
        corpus,
        corpus_prefix=args.corpus_prefix,
        allow_extra_images=args.allow_extra_images,
    )
    if after != state:
        raise ValueError("Source changed while building the control index")
    checks = compare_controls(args.index_dir, target, args.collection)
    save_json(args.output / "controls.json", checks)
    return {"index": str(target), "result": asdict(result), "controls": checks}


def compare_controls(source_dir: Path, target_dir: Path, collection: str) -> dict:
    source, target = Catalog(source_dir, readonly=True), Catalog(target_dir, readonly=True)
    try:
        _, a = source.snapshot(collection)
        _, b = target.snapshot(collection)
        left, right = load_manifest(source, a), load_manifest(target, b)
        _, semantic_a = matrices(source, left)
        _, semantic_b = matrices(target, right)
        checks = {
            "source_generation": a["id"],
            "target_generation": b["id"],
            "images": len(left["images"]),
            "same_image_order": left["vision_ids"] == right["vision_ids"],
            "same_members_and_statuses": left["images"] == right["images"],
            "identical_ocr": left["ocr"] == right["ocr"],
            "same_passage_order": left["passage_ids"] == right["passage_ids"],
            "identical_e5_vectors": np.array_equal(semantic_a, semantic_b),
            "spec_differs_only_in_precision": dict(left["config"]["spec"], siglip_precision="int8")
            == right["config"]["spec"],
        }
        if not all(
            value
            for key, value in checks.items()
            if key not in {"source_generation", "target_generation", "images"}
        ):
            raise ValueError("Controlled precision comparison differs outside SigLIP2")
        return checks
    finally:
        source.close()
        target.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--collection", default="personal")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--corpus-prefix", type=Path, default=Path("."))
    parser.add_argument("--allow-extra-images", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enforce()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
