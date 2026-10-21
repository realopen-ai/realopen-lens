"""Named subsets of one published source collection, keyed by content identity."""

from dataclasses import asdict
import hashlib
from pathlib import Path

from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog
from realopen_lens.storage.organization import OrganizationStore, CollectionConflict, image_ids


def parent(root: Path, collection: str) -> dict:
    catalog = Catalog(root, readonly=True)
    try:
        found, _ = catalog.snapshot(collection)
        return found
    finally:
        catalog.close()


def resolve_scope(
    root: Path, collection_id: str, virtual_id: str, revision: int | None = None
) -> tuple[set[str], dict]:
    if not (root / "organization.sqlite3").is_file():
        raise ValueError("Virtual collection not found")
    with OrganizationStore(root, readonly=True) as store:
        row = store.snapshot(virtual_id, collection_id, revision)
    ids = set(row["image_ids"])
    scope = {key: row[key] for key in ("id", "collection_id", "name", "revision")}
    scope.update(
        member_count=len(ids),
        members_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
    )
    return ids, scope


def ensure_scope(root: Path, collection_id: str, scope: dict) -> None:
    try:
        _, current = resolve_scope(root, collection_id, scope["id"], scope["revision"])
    except ValueError as error:
        raise CollectionConflict("Virtual collection changed; refresh and retry") from error
    if current != scope:
        raise CollectionConflict("Virtual collection changed; refresh and retry")


def list_collections(root: Path, collection: str) -> list[dict]:
    if not (root / "organization.sqlite3").is_file():
        return []
    from realopen_lens.retrieval.search import source_paths

    catalog = Catalog(root, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
        available = set(source_paths(Path(found["root"]), manifest))
        with OrganizationStore(root, readonly=True) as store:
            store.db.execute("BEGIN")
            rows = store.list_collections(found["id"])
            members: dict[str, set[str]] = {}
            for member in store.db.execute(
                "SELECT m.virtual_id,m.image_id FROM members m JOIN virtual_collections v "
                "ON v.id=m.virtual_id WHERE v.collection_id=?",
                (found["id"],),
            ):
                members.setdefault(member["virtual_id"], set()).add(member["image_id"])
            for row in rows:
                row["available_count"] = len(members.get(row["id"], set()) & available)
                row["unavailable_count"] = row["member_count"] - row["available_count"]
            return rows
    finally:
        catalog.close()


def create(root: Path, collection: str, name: str) -> dict:
    found = parent(root, collection)
    with OrganizationStore(root) as store:
        return store.create(found["id"], name)


def rename(root: Path, collection: str, virtual_id: str, name: str, revision: int) -> dict:
    found = parent(root, collection)
    with OrganizationStore(root) as store:
        return store.rename(virtual_id, found["id"], name, revision)


def remove(root: Path, collection: str, virtual_id: str, revision: int) -> None:
    found = parent(root, collection)
    with OrganizationStore(root) as store:
        store.remove(virtual_id, found["id"], revision)


def change_members(
    root: Path,
    collection: str,
    virtual_id: str,
    values: list[str],
    revision: int,
    *,
    remove: bool = False,
) -> dict:
    from realopen_lens.retrieval.search import source_paths

    values = image_ids(values)
    catalog = Catalog(root, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        if not remove:
            manifest = load_manifest(catalog, generation)
            available = source_paths(Path(found["root"]), manifest)
            if not set(values) <= set(available):
                raise ValueError("Only available images in this source collection can be added")
        with OrganizationStore(root) as store:
            return store.members(virtual_id, found["id"], values, revision, remove=remove)
    finally:
        catalog.close()


def browse(
    root: Path,
    collection: str,
    virtual_id: str,
    offset: int = 0,
    limit: int = 60,
    revision: int | None = None,
    generation: str | None = None,
) -> dict:
    from realopen_lens.retrieval.search import source_paths, search_results

    if not 0 <= offset <= 1_000_000 or not 1 <= limit <= 100:
        raise ValueError("Invalid page range")
    catalog = Catalog(root, readonly=True)
    try:
        found, active = catalog.snapshot(collection)
        if generation is not None and active["id"] != generation:
            raise CollectionConflict("Source index changed; refresh the virtual collection")
        ids, scope = resolve_scope(root, found["id"], virtual_id, revision)
        manifest = load_manifest(catalog, active)
        paths = source_paths(Path(found["root"]), manifest)
        available = sorted(ids & paths.keys())
        records = search_results(catalog, manifest, paths, available[offset : offset + limit], {})
        ensure_scope(root, found["id"], scope)
        return {
            "results": [asdict(r) for r in records],
            "total": len(available),
            "member_count": len(ids),
            "unavailable_count": len(ids) - len(available),
            "generation": active["id"],
            "scope": scope,
        }
    finally:
        catalog.close()
