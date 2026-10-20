"""Validate a frozen corpus against a production collection snapshot."""

from pathlib import Path
from benchmarks.datasets.downloads import digest
from realopen_lens.indexing.service import signature
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog


def snapshot(
    index_dir: Path,
    collection: str,
    corpus: dict,
    verify_files: bool = True,
    *,
    corpus_prefix: Path = Path("."),
    allow_extra_images: bool = False,
) -> dict:
    if corpus_prefix.is_absolute() or ".." in corpus_prefix.parts:
        raise ValueError("Corpus prefix must stay relative to the collection root")
    catalog = Catalog(index_dir, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
        expected = {r["sha256"]: r for r in corpus["images"]}
        observed = {r["image_id"]: r for r in manifest["images"]}
        if not set(expected) <= set(observed):
            raise ValueError("Saved generation is missing frozen corpus images")
        if not allow_extra_images and set(expected) != set(observed):
            raise ValueError("Saved generation must contain exactly the frozen unique corpus")
        signatures = {}
        for key, row in observed.items():
            paths = row["paths"]
            if not paths:
                raise ValueError("Published image has no source paths")
            if key in expected:
                relative = (corpus_prefix / expected[key]["path"]).as_posix()
                if relative not in row["paths"]:
                    raise ValueError(f"Frozen path is absent from saved collection: {relative}")
                if not allow_extra_images:
                    paths = [relative]
            # In superset mode every alias and unreviewed distractor is frozen too.
            checked = {}
            for relative in paths:
                path = (Path(found["root"]) / relative).resolve()
                if not path.is_relative_to(Path(found["root"]).resolve()):
                    raise ValueError("Corpus path escapes the collection")
                if signature(path) != row["signatures"][relative] or (
                    verify_files and digest(path) != key
                ):
                    raise ValueError(f"Source changed; index again: {relative}")
                checked[relative] = row["signatures"][relative]
            signatures[key] = checked if allow_extra_images else checked[paths[0]]
        state = {
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
        if allow_extra_images or corpus_prefix != Path("."):
            state.update(
                corpus_prefix=corpus_prefix.as_posix(),
                allow_extra_images=allow_extra_images,
                reference_images=len(expected),
                additional_unreviewed_images=len(set(observed) - set(expected)),
            )
        return state
    finally:
        catalog.close()
