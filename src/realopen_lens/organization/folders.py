"""Folder identity changes; original files and shared caches are never mutated."""

import json
from pathlib import Path
import unicodedata

from realopen_lens.runtime.jobs import writer_lock
from realopen_lens.storage.catalog import Catalog
from realopen_lens.storage.organization import CollectionConflict


def checked(catalog: Catalog, name: str, collection_id: str) -> dict:
    row = catalog.db.execute("SELECT * FROM collections WHERE name=?", (name,)).fetchone()
    if row is None or row["id"] != collection_id:
        raise CollectionConflict("Folder changed or was removed; reload the folder list")
    return dict(row)


def refresh_options(root: Path, name: str, collection_id: str) -> dict:
    catalog = Catalog(root, readonly=True)
    try:
        found = checked(catalog, name, collection_id)
        # An interrupted first index has no generation; use its latest job settings.
        job = catalog.db.execute(
            "SELECT config FROM jobs WHERE collection_id=? ORDER BY started_at DESC LIMIT 1",
            (collection_id,),
        ).fetchone()
        config = json.loads(job["config"]) if job else {}
        if found["active_generation"]:
            row = catalog.db.execute(
                "SELECT j.config FROM generations g JOIN jobs j ON j.id=g.job_id WHERE g.id=?",
                (found["active_generation"],),
            ).fetchone()
            config = json.loads(row["config"])
        path = Path(found["root"])
        if not path.is_dir():
            raise ValueError("Source folder is unavailable; reconnect it before refreshing")
        spec = config.get("spec", config)
        return {
            "root": str(path),
            "collection": name,
            "languages": spec.get("languages", "eng+fra+ara"),
            "siglip_precision": spec.get("siglip_precision", "int8"),
        }
    finally:
        catalog.close()


def change(root: Path, name: str, collection_id: str, new_name: str | None) -> dict:
    """Rename in place or unregister a folder under the same lock as indexing."""
    if new_name is not None:
        new_name = unicodedata.normalize("NFKC", new_name).strip()
        if not 1 <= len(new_name) <= 128 or any(
            unicodedata.category(c).startswith("C") for c in new_name
        ):
            raise ValueError("Use a folder name of 1–128 characters without control characters")
    with writer_lock(root):
        catalog = Catalog(root)
        try:
            with catalog.transaction():
                found = checked(catalog, name, collection_id)
                if new_name is not None:
                    taken = catalog.db.execute(
                        "SELECT id FROM collections WHERE name=? AND id!=?",
                        (new_name, collection_id),
                    ).fetchone()
                    if taken:
                        raise CollectionConflict("That folder name is already in use")
                # Pin legacy name-based model settings to the stable ID before rename/removal.
                # This remains valid even if the catalog transaction subsequently fails.
                from realopen_lens.learning.registry import pin_collection

                pin_collection(root, name, collection_id)
                if new_name is None:
                    catalog.db.execute(
                        "DELETE FROM files WHERE job_id IN "
                        "(SELECT id FROM jobs WHERE collection_id=?)",
                        (collection_id,),
                    )
                    catalog.db.execute(
                        "DELETE FROM generations WHERE collection_id=?", (collection_id,)
                    )
                    catalog.db.execute("DELETE FROM jobs WHERE collection_id=?", (collection_id,))
                    catalog.db.execute("DELETE FROM collections WHERE id=?", (collection_id,))
                else:
                    catalog.db.execute(
                        "UPDATE collections SET name=? WHERE id=?", (new_name, collection_id)
                    )
            return {"id": found["id"], "name": new_name, "removed": new_name is None}
        finally:
            catalog.close()
