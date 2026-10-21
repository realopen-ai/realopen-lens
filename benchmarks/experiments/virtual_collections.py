"""Native virtual-collection checks on an isolated copy of a published index."""

import argparse
from dataclasses import asdict
import json
from importlib.metadata import version
import platform
import realopen_lens
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter

from PIL import Image, ImageOps

from realopen_lens.learning.dataset import canonical
from realopen_lens.models.manager import digest
from realopen_lens.organization import collections as organizing
from realopen_lens.retrieval.images import search_similar
from realopen_lens.retrieval.search import source_paths
from realopen_lens.retrieval.session import SearchSession
from realopen_lens.storage.artifacts import atomic_bytes, load_manifest
from realopen_lens.storage.catalog import Catalog, local_path


def copy_snapshot(source: Path, output: Path, collection: str) -> dict:
    """Copy immutable published artifacts, without copying feedback or deployment state."""
    output.mkdir(parents=True, exist_ok=False)
    original = Catalog(source, readonly=True)
    try:
        with sqlite3.connect(output / "catalog.sqlite3") as destination:
            original.db.backup(destination)
    finally:
        original.close()
    catalog = Catalog(output, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        base = Path(generation["manifest"]).parent
        shutil.copytree(local_path(source, base.as_posix()), local_path(output, base.as_posix()))
        manifest = load_manifest(catalog, generation)
        for row in manifest["images"]:
            path = Path(row["thumbnail"])
            target = local_path(output, path.as_posix())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path(source, path.as_posix()), target)
        return {
            "collection": found["name"],
            "generation": generation["id"],
            "manifest_sha256": generation["checksum"],
            "precision": manifest["config"]["spec"].get("siglip_precision", "int8"),
            "indexed_images": len(manifest["images"]),
            "pipeline_spec": manifest["config"]["spec"],
        }
    finally:
        catalog.close()


def run(source: Path, output: Path, collection: str, models: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    index = output / "index"
    identity = copy_snapshot(source, index, collection)
    package = Path(realopen_lens.__file__).parent
    identity["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": 2,
        "packages": {
            name: version(name)
            for name in ("numpy", "onnxruntime", "Pillow", "psutil", "tokenizers")
        },
    }
    identity["source_sha256"] = {
        name: digest(package / name)
        for name in (
            "organization/collections.py",
            "storage/organization.py",
            "retrieval/search.py",
            "retrieval/images.py",
            "learning/feedback.py",
        )
    }
    atomic_bytes(output / "config.json", canonical(identity))
    catalog = Catalog(index, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
        paths = source_paths(Path(found["root"]), manifest)
    finally:
        catalog.close()
    if len(paths) < 2:
        raise ValueError("Native check needs at least two available indexed images")
    ids = sorted(paths)
    group = organizing.create(index, collection, "Saved selection")
    group = organizing.change_members(index, collection, group["id"], ids[:61], 1)
    empty = organizing.create(index, collection, "Empty collection")
    records = []
    with (output / "cases.jsonl").open("x", encoding="utf-8", newline="\n") as stream:

        def case(name, operation):
            started = perf_counter()
            row = {"case": name}
            try:
                row.update(operation())
                row["status"] = "ok"
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            row["seconds"] = perf_counter() - started
            records.append(row)
            stream.write(canonical(row).decode("utf-8") + "\n")
            stream.flush()
            print(json.dumps({"case": name, "status": row["status"]}), flush=True)
            return row

        def pages():
            first = organizing.browse(index, collection, group["id"], limit=60)
            next_page = organizing.browse(
                index,
                collection,
                group["id"],
                offset=60,
                revision=group["revision"],
                generation=first["generation"],
            )
            observed = [r["image_id"] for r in first["results"] + next_page["results"]]
            assert observed == ids[:61]
            return {"count": len(observed), "revision": group["revision"]}

        case("paged_membership", pages)
        with SearchSession(collection=collection, index_dir=index, models=models) as session:
            baseline_trace: dict = {}
            baseline = session.search("pizza", limit=50, trace=baseline_trace)
            if len(paths) > 50:
                hidden = next(key for key in ids if key not in {r.image_id for r in baseline})
                only = organizing.create(index, collection, "Outside global top 50")
                only = organizing.change_members(index, collection, only["id"], [hidden], 1)

                def below50():
                    results = session.search("pizza", virtual_collection=only["id"], limit=50)
                    assert [r.image_id for r in results] == [hidden]
                    return {"expected": hidden, "results": [asdict(r) for r in results]}

                case("pre_candidate_filter", below50)
            for mode in ("hybrid", "visual", "semantic", "lexical"):

                def query(mode=mode):
                    trace: dict = {}
                    options = {"trace": trace} if mode == "hybrid" else {}
                    results = session.search(
                        "pizza", mode=mode, limit=20, virtual_collection=group["id"], **options
                    )
                    assert {r.image_id for r in results} <= set(ids[:61])
                    return {"mode": mode, "results": [asdict(r) for r in results], "trace": trace}

                case(f"text_{mode}", query)

            def unchanged():
                trace: dict = {}
                results = session.search("pizza", limit=50, trace=trace)
                assert results == baseline and trace == baseline_trace
                return {"count": len(results), "exact_trace_match": True}

            case("unscoped_unchanged", unchanged)

        def no_models():
            with SearchSession(collection=collection, index_dir=index, models=models) as session:
                assert session.search("pizza", virtual_collection=empty["id"]) == []
                assert session.worker_pid is None
            return {"result_count": 0, "spawned_encoder": False}

        case("empty_scope", no_models)

        def similar():
            trace: dict = {}
            results = search_similar(
                image_id=ids[-1],
                collection=collection,
                index_dir=index,
                virtual_collection=group["id"],
                trace=trace,
            )
            assert {r.image_id for r in results} <= set(ids[:61])
            assert all(r.image_id != ids[-1] for r in results)
            assert trace["reused_embedding"]
            return {"results": [asdict(r) for r in results], "trace": trace}

        case("indexed_image_similarity", similar)
        query_path = output / "external-query.png"
        with Image.open(paths[ids[0]][0]) as im:
            sample = ImageOps.contain(im.convert("RGB"), (320, 320))
            canvas = Image.new("RGB", (341, 337), "white")
            canvas.paste(sample, (0, 0))
            canvas.save(query_path)

        def external():
            trace: dict = {}
            results = search_similar(
                query_path,
                collection=collection,
                index_dir=index,
                models=models,
                virtual_collection=group["id"],
                trace=trace,
            )
            assert {r.image_id for r in results} <= set(ids[:61])
            assert not trace["reused_embedding"]
            return {
                "query_sha256": digest(query_path),
                "results": [asdict(r) for r in results],
                "trace": trace,
            }

        case("external_image_similarity", external)
    summary = {
        "purpose": "functional subset/identity validation; no new relevance judgments",
        **identity,
        "available_images": len(ids),
        "cases": len(records),
        "failed": sum(r["status"] != "ok" for r in records),
        "group_id": group["id"],
        "empty_group_id": empty["id"],
    }
    atomic_bytes(output / "summary.json", canonical(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection", default="personal")
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    result = run(args.source_index, args.output, args.collection, args.models)
    raise SystemExit(1 if result["failed"] else 0)
