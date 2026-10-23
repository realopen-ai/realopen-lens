"""Versioned SQLite virtual collections; no source-file or index mutations."""

from pathlib import Path
import re
import sqlite3
import unicodedata

from realopen_lens.storage.catalog import identifier, now


class CollectionConflict(ValueError):
    """An operation refers to an older membership/name revision."""


def name_parts(name: str) -> tuple[str, str]:
    name = unicodedata.normalize("NFKC", name).strip()
    if not 1 <= len(name) <= 80 or any(unicodedata.category(c).startswith("C") for c in name):
        raise ValueError("Use a collection name of 1–80 characters without control characters")
    return name, name.casefold()


def image_ids(values: list[str]) -> list[str]:
    if not 1 <= len(values) <= 500 or any(
        not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in values
    ):
        raise ValueError("Supply 1–500 indexed image SHA256 IDs")
    return sorted(set(values))


class OrganizationStore:
    def __init__(self, root: Path, *, readonly: bool = False):
        path = root.resolve() / "organization.sqlite3"
        if not readonly:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"), uri=True, timeout=30
        )
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA foreign_keys=ON")
            if not readonly:
                # Check ownership and initialize under the same SQLite write lock.
                # A second first-use request must observe the committed version.
                self.db.execute("BEGIN IMMEDIATE")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1) or (readonly and version == 0):
                raise ValueError("Unsupported organization database version")
            if not readonly:
                if version == 0:
                    if self.db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    ).fetchone():
                        raise ValueError(
                            "Refusing to initialize an unrelated organization database"
                        )
                    statements = (
                        "CREATE TABLE virtual_collections ("
                        "id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, "
                        "name TEXT NOT NULL, name_key TEXT NOT NULL, "
                        "revision INTEGER NOT NULL CHECK(revision > 0), "
                        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                        "UNIQUE(collection_id, name_key))",
                        "CREATE TABLE members (virtual_id TEXT NOT NULL "
                        "REFERENCES virtual_collections(id) ON DELETE CASCADE, "
                        "image_id TEXT NOT NULL, added_at TEXT NOT NULL, "
                        "PRIMARY KEY(virtual_id,image_id))",
                        "CREATE INDEX virtual_parent ON virtual_collections(collection_id)",
                        "PRAGMA user_version=1",
                    )
                    for statement in statements:
                        self.db.execute(statement)
                self.db.commit()
                try:
                    self.db.execute("PRAGMA journal_mode=WAL")
                except sqlite3.OperationalError as error:
                    # Existing rollback journaling is also safe; another writer can
                    # briefly prevent the optional switch to WAL on first use.
                    if error.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                        raise
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def get(self, virtual_id: str, collection_id: str, revision: int | None = None) -> dict:
        row = self.db.execute(
            "SELECT * FROM virtual_collections WHERE id=? AND collection_id=?",
            (virtual_id, collection_id),
        ).fetchone()
        if row is None:
            raise ValueError("Virtual collection not found in this source collection")
        if revision is not None and (type(revision) is not int or row["revision"] != revision):
            raise CollectionConflict("Virtual collection changed; refresh and retry")
        return dict(row)

    def snapshot(self, virtual_id: str, collection_id: str, revision: int | None = None) -> dict:
        with self.db:
            self.db.execute("BEGIN")
            result = self.get(virtual_id, collection_id, revision)
            result["image_ids"] = [
                r[0]
                for r in self.db.execute(
                    "SELECT image_id FROM members WHERE virtual_id=? ORDER BY image_id",
                    (virtual_id,),
                )
            ]
            return result

    def list_collections(self, collection_id: str) -> list[dict]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT v.*,COUNT(m.image_id) AS member_count FROM virtual_collections v "
                "LEFT JOIN members m ON m.virtual_id=v.id WHERE v.collection_id=? "
                "GROUP BY v.id ORDER BY v.name_key,v.id",
                (collection_id,),
            )
        ]

    def create(self, collection_id: str, name: str) -> dict:
        name, key = name_parts(name)
        stamp, virtual_id = now(), identifier()
        try:
            with self.db:
                self.db.execute(
                    "INSERT INTO virtual_collections VALUES (?,?,?,?,1,?,?)",
                    (virtual_id, collection_id, name, key, stamp, stamp),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("A virtual collection with that name already exists") from error
        return self.get(virtual_id, collection_id)

    def rename(self, virtual_id: str, collection_id: str, name: str, revision: int) -> dict:
        name, key = name_parts(name)
        try:
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                old = self.get(virtual_id, collection_id, revision)
                if old["name"] != name:
                    self.db.execute(
                        "UPDATE virtual_collections SET name=?,name_key=?,"
                        "revision=revision+1,updated_at=? WHERE id=?",
                        (name, key, now(), virtual_id),
                    )
        except sqlite3.IntegrityError as error:
            raise ValueError("A virtual collection with that name already exists") from error
        return self.get(virtual_id, collection_id)

    def remove(self, virtual_id: str, collection_id: str, revision: int) -> None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.get(virtual_id, collection_id, revision)
            self.db.execute("DELETE FROM virtual_collections WHERE id=?", (virtual_id,))

    def members(
        self,
        virtual_id: str,
        collection_id: str,
        values: list[str],
        revision: int,
        *,
        remove: bool = False,
        resolved_values: list[str] | None = None,
    ) -> dict:
        values = image_ids(values)
        if resolved_values is not None:
            # An internal alias expansion may remove more stored IDs than the
            # 500 logical images accepted by an adapter. Keep one transaction.
            values = sorted(set(resolved_values))
            if not values or len(values) > 100_000:
                raise ValueError("Invalid resolved membership batch")
            for start in range(0, len(values), 500):
                image_ids(values[start : start + 500])
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.get(virtual_id, collection_id, revision)
            before = self.db.total_changes
            if remove:
                self.db.executemany(
                    "DELETE FROM members WHERE virtual_id=? AND image_id=?",
                    [(virtual_id, value) for value in values],
                )
            else:
                self.db.executemany(
                    "INSERT OR IGNORE INTO members VALUES (?,?,?)",
                    [(virtual_id, value, now()) for value in values],
                )
            if self.db.total_changes != before:
                self.db.execute(
                    "UPDATE virtual_collections SET revision=revision+1,updated_at=? " "WHERE id=?",
                    (now(), virtual_id),
                )
        return self.get(virtual_id, collection_id)
