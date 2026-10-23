"""Workspace-wide virtual collections, keyed by exact image content identities."""

from pathlib import Path

from realopen_lens.organization import duplicates
from realopen_lens.retrieval.search import source_paths
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog
from realopen_lens.storage.organization import OrganizationStore, image_ids

LIBRARY_ID = "library-v1"


def available(root: Path) -> tuple[set[str], dict[str, str]]:
    """Read published source identities once; never infer, decode, or alter images."""
    if not (root / "catalog.sqlite3").is_file():
        return set(), {}
    catalog = Catalog(root, readonly=True)
    try:
        catalog.db.execute("BEGIN")
        ids: set[str] = set()
        pixels: dict[str, str] = {}
        for folder in catalog.db.execute("SELECT * FROM collections ORDER BY id"):
            if not folder["active_generation"]:
                continue
            generation = dict(
                catalog.db.execute(
                    "SELECT * FROM generations WHERE id=?", (folder["active_generation"],)
                ).fetchone()
            )
            manifest = load_manifest(catalog, generation)
            ids.update(source_paths(Path(folder["root"]), manifest))
            cached = duplicates.snapshot(root, manifest)
            if cached:
                pixels.update(cached.get("pixels", {}))
        return ids, pixels
    finally:
        catalog.close()


def match_members(
    root: Path, members: set[str], canonical: dict[str, str], pixels: dict[str, str]
) -> tuple[set[str], int]:
    """Resolve saved aliases, including a removed format with a surviving pixel copy."""
    by_pixels = {value: canonical[key] for key, value in pixels.items() if key in canonical}
    groups = set()
    missing = 0
    for key in members:
        if key in canonical:
            groups.add(canonical[key])
            continue
        cached = duplicates._cached(root / "duplicates/cache" / (key + ".json"), key)
        pixel = pixels.get(key) or (cached or {}).get("pixels")
        if pixel in by_pixels:
            groups.add(by_pixels[pixel])
        else:
            missing += 1
    return groups, missing


def list_collections(root: Path) -> list[dict]:
    if not (root / "organization.sqlite3").is_file():
        return []
    ids, pixels = available(root)
    canonical = {key: key for key in ids}
    with OrganizationStore(root, readonly=True) as store:
        store.db.execute("BEGIN")
        rows = store.list_collections(LIBRARY_ID)
        memberships: dict[str, set[str]] = {}
        for row in store.db.execute(
            "SELECT m.virtual_id,m.image_id FROM members m JOIN virtual_collections v "
            "ON v.id=m.virtual_id WHERE v.collection_id=?",
            (LIBRARY_ID,),
        ):
            memberships.setdefault(row["virtual_id"], set()).add(row["image_id"])
        for row in rows:
            _, missing = match_members(root, memberships.get(row["id"], set()), canonical, pixels)
            row.update(available_count=row["member_count"] - missing, unavailable_count=missing)
        return rows


def change_members(
    root: Path, virtual_id: str, values: list[str], revision: int, *, remove: bool = False
) -> dict:
    values = image_ids(values)
    ids, pixels = available(root)
    if not remove and not set(values) <= ids:
        raise ValueError("Only available images in this library can be added")

    def identity(key: str) -> str:
        cached = duplicates._cached(root / "duplicates/cache" / (key + ".json"), key)
        pixel = pixels.get(key) or (cached or {}).get("pixels")
        return "pixels:" + pixel if pixel else "bytes:" + key

    with OrganizationStore(root) as store:
        old = store.snapshot(virtual_id, LIBRARY_ID, revision)
        previous = set(old["image_ids"])
        if remove:
            selected = {identity(key) for key in values}
            resolved = sorted(key for key in previous if identity(key) in selected)
        else:
            seen = {identity(key) for key in previous}
            resolved = []
            for key in values:
                content = identity(key)
                if content not in seen:
                    resolved.append(key)
                    seen.add(content)
        if not resolved:
            return store.get(virtual_id, LIBRARY_ID, revision)
        return store.members(
            virtual_id, LIBRARY_ID, values, revision, remove=remove, resolved_values=resolved
        )
