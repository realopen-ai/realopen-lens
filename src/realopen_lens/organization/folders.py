"""Folder identity changes; original files and shared caches are never mutated."""

import json
from pathlib import Path

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
