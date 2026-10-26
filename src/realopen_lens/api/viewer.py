"""On-demand indexed image details and explicit local desktop actions."""

from functools import cache
from pathlib import Path
import subprocess
import sys

from realopen_lens.api.collections import snapshot
from realopen_lens.models.manager import digest
from realopen_lens.indexing.scanner import EXTENSIONS
from realopen_lens.retrieval.search import source_paths
from realopen_lens.storage.catalog import Catalog
from realopen_lens.storage.organization import OrganizationStore


def resolve(catalog: Catalog, collection: str, collection_id: str | None) -> str:
    return catalog.collection_name(collection_id) if collection_id else collection


def original(root: Path, collection: str, image_id: str, generation: str | None) -> Path:
    """Resolve only unchanged, content-verified indexed raster originals."""
    catalog = Catalog(root, readonly=True)
    try:
        found, manifest = snapshot(catalog, collection, generation)
        row = next((r for r in manifest["images"] if r["image_id"] == image_id), None)
        if row is None:
            raise ValueError("Image not found")
        paths = source_paths(Path(found["root"]), {"images": [row]}).get(image_id, ())
        for value in paths:
            path = Path(value)
            if path.suffix.lower() in EXTENSIONS and digest(path) == image_id:
                return path
        raise ValueError("Original image changed or is unavailable; refresh the folder index")
    finally:
        catalog.close()


def memberships(root: Path, collection_id: str, image_id: str) -> list[dict]:
    if not (root / "organization.sqlite3").is_file():
        return []
    from realopen_lens.organization.duplicates import _cached

    @cache
    def identity(key: str) -> str:
        cached = _cached(root / "duplicates/cache" / (key + ".json"), key)
        return "pixels:" + cached["pixels"] if cached and cached.get("pixels") else key

    target = identity(image_id)
    with OrganizationStore(root, readonly=True) as store:
        store.db.execute("BEGIN")
        groups = [*store.list_collections(collection_id), *store.list_collections("library-v1")]
        for group in groups:
            group["member_image_ids"] = [
                row[0]
                for row in store.db.execute(
                    "SELECT image_id FROM members WHERE virtual_id=?", (group["id"],)
                )
                if identity(row[0]) == target
            ]
        return groups


def details(root: Path, collection: str, image_id: str, generation: str | None = None) -> dict:
    catalog = Catalog(root, readonly=True)
    try:
        found, manifest = snapshot(catalog, collection, generation)
        row = next((r for r in manifest["images"] if r["image_id"] == image_id), None)
        if row is None:
            raise ValueError("Image not found")
        from realopen_lens.ocr.languages import selection

        ocr: dict = next((r for r in manifest["ocr"] if r["image_id"] == image_id), {})
        text = ocr.get("text", "")
        paths = source_paths(Path(found["root"]), {"images": [row]}).get(image_id, ())
        language_selection = ocr.get("language_selection") or selection(
            {}, manifest["config"]["spec"].get("languages")
        )
        return {
            "image_id": image_id,
            "collection": found["name"],
            "collection_id": found["id"],
            "generation": generation or found["active_generation"],
            "metadata": row["metadata"],
            "paths": [str(Path(found["root"]) / path) for path in row["paths"]],
            "available_paths": list(paths),
            "text": text,
            "ocr_language_selection": language_selection,
            "issues": row.get("issues", []),
            "groups": memberships(root, found["id"], image_id),
        }
    finally:
        catalog.close()


def desktop_action(path: Path, action: str) -> None:
    """Use argument arrays, never a shell; only the caller's verified image is accepted."""
    if sys.platform == "darwin":
        command = ["/usr/bin/open", *(["-R"] if action == "reveal" else []), str(path)]
    elif sys.platform == "win32":
        import os

        if action == "open":
            os.startfile(str(path))
            return
        command = ["explorer.exe", "/select,", str(path)]
    elif sys.platform.startswith("linux"):
        command = ["xdg-open", str(path.parent if action == "reveal" else path)]
    else:
        raise ValueError("Desktop actions are unavailable on this platform")
    try:
        subprocess.run(command, check=True, timeout=10, capture_output=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "Could not open the local image or folder; check your desktop session"
        ) from error
