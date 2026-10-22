"""Read-only collection browsing and ID-based media resolution."""

from dataclasses import asdict
from collections import OrderedDict
import threading
import json
from pathlib import Path

from realopen_lens.retrieval.search import source_paths, search_results
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog, local_path
from realopen_lens.indexing.service import signature

_manifests: OrderedDict[tuple, tuple[dict, tuple]] = OrderedDict()
_manifest_lock = threading.Lock()


def _manifest(catalog: Catalog, active: dict) -> dict:
    """Retain at most two verified immutable snapshots; filesystem changes invalidate."""
    key = (str(catalog.root), active["id"], active["checksum"])

    def signatures(manifest: dict) -> tuple:
        paths = [active["manifest"], *(a["path"] for a in manifest["artifacts"].values())]
        return tuple(tuple(signature(local_path(catalog.root, p)).items()) for p in paths)

    with _manifest_lock:
        cached = _manifests.get(key)
        if cached is not None and cached[1] == signatures(cached[0]):
            _manifests.move_to_end(key)
            return cached[0]
        manifest = load_manifest(catalog, active)
        before = signatures(manifest)
        # Recheck bytes before caching against the captured signatures.
        manifest = load_manifest(catalog, active)
        if before != signatures(manifest):
            raise ValueError("Index changed while validating gallery snapshot")
        _manifests[key] = (manifest, before)
        while len(_manifests) > 2:
            _manifests.popitem(last=False)
        return manifest


def collections(root: Path) -> list[dict]:
    if not (root / "catalog.sqlite3").is_file():
        return []
    catalog = Catalog(root, readonly=True)
    try:
        result = []
        for row in catalog.db.execute("SELECT * FROM collections ORDER BY name"):
            spec = {}
            if row["active_generation"]:
                generation = catalog.db.execute(
                    "SELECT job_id FROM generations WHERE id=?", (row["active_generation"],)
                ).fetchone()
                spec = json.loads(catalog.job(generation["job_id"])["config"])["spec"]
            result.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "root": row["root"],
                    "generation": row["active_generation"],
                    "siglip_precision": spec.get("siglip_precision", "int8"),
                }
            )
        return result
    finally:
        catalog.close()


def snapshot(catalog: Catalog, collection: str, generation: str | None = None) -> tuple[dict, dict]:
    found, active = catalog.snapshot(collection)
    if generation is not None:
        row = catalog.db.execute(
            "SELECT * FROM generations WHERE id=? AND collection_id=?", (generation, found["id"])
        ).fetchone()
        if row is None:
            raise ValueError("Collection generation not found")
        active = dict(row)
    return found, _manifest(catalog, active)


def browse(root: Path, collection: str, offset: int, limit: int) -> dict:
    catalog = Catalog(root, readonly=True)
    try:
        found, manifest = snapshot(catalog, collection)
        paths = source_paths(Path(found["root"]), manifest)
        ids = [r["image_id"] for r in manifest["images"] if r["image_id"] in paths]
        records = search_results(catalog, manifest, paths, ids[offset : offset + limit], {})
        return {
            "results": [asdict(r) for r in records],
            "total": len(ids),
            "generation": found["active_generation"],
        }
    finally:
        catalog.close()


def media(
    root: Path, collection: str, image_id: str, thumbnail: bool, generation: str | None = None
) -> tuple[bytes, str]:
    catalog = Catalog(root, readonly=True)
    try:
        found, manifest = snapshot(catalog, collection, generation)
        row = next((r for r in manifest["images"] if r["image_id"] == image_id), None)
        if row is None:
            raise ValueError("Image not found")
        if thumbnail:
            path = local_path(root, row["thumbnail"])
            size, expected = 4 * 1024 * 1024, row["thumbnail_sha256"]
        else:
            paths = source_paths(Path(found["root"]), {"images": [row]})
            if image_id not in paths:
                raise ValueError("Original image changed or is unavailable")
            path = Path(paths[image_id][0])
            size, expected = 64 * 1024 * 1024, image_id
        with path.open("rb") as stream:
            data = stream.read(size + 1)
        import hashlib

        if len(data) > size or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Image is missing, changed or corrupt")
        # Only indexed raster data is served, never arbitrary user-provided paths.
        import mimetypes

        return data, (
            "image/jpeg"
            if thumbnail
            else (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        )
    finally:
        catalog.close()
