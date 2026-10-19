"""Validate a frozen corpus against a production collection snapshot."""

from pathlib import Path
from benchmarks.datasets.downloads import digest
from realopen_lens.indexing.service import signature
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog


def snapshot(index_dir: Path, collection: str, corpus: dict, verify_files: bool = True) -> dict:
    catalog = Catalog(index_dir, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
        expected = {r["sha256"]: r for r in corpus["images"]}
        observed = {r["image_id"]: r for r in manifest["images"]}
        if set(expected) != set(observed):
            raise ValueError("Saved generation must contain exactly the frozen unique corpus")
        signatures = {}
        for key, row in observed.items():
            relative = expected[key]["path"]
            if relative not in row["paths"]:
                raise ValueError(f"Frozen path is absent from saved collection: {relative}")
            path = (Path(found["root"]) / relative).resolve()
            if not path.is_relative_to(Path(found["root"]).resolve()):
                raise ValueError("Corpus path escapes the collection")
            if signature(path) != row["signatures"][relative] or (
                verify_files and digest(path) != key
            ):
                raise ValueError(f"Source changed; index again: {relative}")
            signatures[key] = row["signatures"][relative]
        return {
            "generation": generation["id"],
            "manifest_sha256": generation["checksum"],
            "pipeline": manifest["pipeline"],
            "pipeline_spec": manifest["config"]["spec"],
            "root": found["root"],
            "unique_images": len(observed),
            "paths": sum(len(r["paths"]) for r in observed.values()),
            "degraded_images": sum(r.get("status") == "degraded" for r in observed.values()),
            "signatures": signatures,
        }
    finally:
        catalog.close()
