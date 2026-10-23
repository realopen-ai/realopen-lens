"""Bounded metadata-only change detection for registered image folders."""

import hashlib
import json
import os
from pathlib import Path
from time import monotonic
from collections.abc import Callable

from realopen_lens.indexing.scanner import EXTENSIONS
from realopen_lens.storage.catalog import Catalog

MAX_ENTRIES = 100_000
SCAN_SECONDS = 10


def source_identity(root: Path) -> list[int]:
    info = root.stat()
    if not root.is_dir():
        raise ValueError("Source folder is unavailable; reconnect it before refreshing")
    return [info.st_dev, info.st_ino]


def check_source(root: Path, expected: list[int] | None):
    if expected is not None and source_identity(root) != expected:
        raise ValueError("Source folder identity changed; review the folder before refreshing")


def fingerprint(rows: list[tuple[str, dict | None]]) -> str:
    return hashlib.sha256(json.dumps(sorted(rows), sort_keys=True).encode("utf-8")).hexdigest()


def indexed(root: Path, collection_id: str) -> str | None:
    catalog = Catalog(root, readonly=True)
    try:
        row = catalog.db.execute(
            "SELECT g.job_id FROM collections c JOIN generations g ON g.id=c.active_generation "
            "WHERE c.id=?",
            (collection_id,),
        ).fetchone()
        if row is None:
            return None
        return fingerprint(
            [
                (r["path"], json.loads(r["signature"]))
                for r in catalog.db.execute(
                    "SELECT path,signature FROM files WHERE job_id=? ORDER BY path",
                    (row["job_id"],),
                )
            ]
        )
    finally:
        catalog.close()


def observed(root: Path, store: Path, expected: list[int], stopped: Callable[[], bool]) -> str:
    from realopen_lens.indexing.service import signature

    check_source(root, expected)
    deadline, entries, rows = monotonic() + SCAN_SECONDS, 0, []

    pending = [root]
    while pending:
        directory = pending.pop()
        # Consume entries incrementally rather than building an unbounded listing.
        with os.scandir(directory) as found:
            for entry in found:
                entries += 1
                if stopped():
                    raise InterruptedError("Folder monitoring stopped")
                if entries > MAX_ENTRIES or monotonic() > deadline:
                    raise ValueError("Automatic scan limit reached; use Refresh index manually")
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if not path.resolve().is_relative_to(root):
                        raise ValueError(
                            "Directory link leaves the monitored root; use manual refresh"
                        )
                    if entry.name != ".git" and not path.resolve().is_relative_to(store):
                        pending.append(path)
                    continue
                if entry.is_symlink() and entry.is_dir():
                    continue  # The production indexer does not follow directory links.
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                resolved = path.resolve()
                # Match the indexer's signature for invalid image links/non-files.
                info = (
                    signature(path)
                    if resolved.is_relative_to(root)
                    and not resolved.is_relative_to(store)
                    and resolved.is_file()
                    else None
                )
                rows.append((path.relative_to(root).as_posix(), info))
    check_source(root, expected)
    return fingerprint(rows)
