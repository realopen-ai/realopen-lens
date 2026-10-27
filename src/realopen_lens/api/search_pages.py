"""Bounded ephemeral search snapshots; continuations never recompute ranking."""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading
from time import monotonic
from typing import Callable

from realopen_lens.indexing.service import signature
from realopen_lens.storage.catalog import Catalog, identifier
from realopen_lens.storage.feedback import FeedbackStore, memory_scope
from realopen_lens.storage.organization import CollectionConflict, OrganizationStore

POOL_LIMIT = 1000
MAX_SNAPSHOTS = 8
TTL_SECONDS = 1800


class SearchPages:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self.snapshots: OrderedDict[str, dict] = OrderedDict()

    def identity(self, request: dict) -> str:
        catalog = Catalog(self.root, readonly=True)
        try:
            if request.get("all_folders"):
                rows = list(catalog.db.execute("SELECT * FROM collections ORDER BY id"))
            else:
                rows = list(
                    catalog.db.execute(
                        "SELECT * FROM collections WHERE name=?", (request["collection"],)
                    )
                )
            scope = None
            if request.get("virtual_collection"):
                if not rows and not request.get("all_folders"):
                    raise CollectionConflict("Search folder changed; run the search again")
                parent = "library-v1" if request.get("all_folders") else rows[0]["id"]
                try:
                    with OrganizationStore(self.root, readonly=True) as store:
                        scope = store.get(
                            request["virtual_collection"], parent, request.get("virtual_revision")
                        )
                except ValueError as error:
                    raise CollectionConflict(
                        "Search scope changed; run the search again"
                    ) from error
            duplicate_files = sorted((self.root / "duplicates").glob("*.json"))
            state = {
                "folders": [dict(row) for row in rows],
                "scope": scope,
                "duplicates": [(path.name, signature(path)) for path in duplicate_files],
            }
            return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        finally:
            catalog.close()

    def create(self, request: dict, retrieve: Callable[[dict], dict]) -> dict:
        identity = self.identity(request)
        head = retrieve({**request, "defer_record": bool(request.get("save_run"))})
        head_files = {
            path: signature(Path(path)) for row in head["results"] for path in row["paths"]
        }
        # Preserve the ordinary first page, including any trained reranker. The
        # continuation is explicitly deeper hybrid retrieval, never learned output.
        tail = retrieve({**request, "limit": POOL_LIMIT, "save_run": False, "deep_search": True})
        results: list[dict] = []
        seen: set[str] = set()
        for row in [*head["results"], *tail["results"]]:
            keys = {row["image_id"], *row.get("duplicate_ids", [])}
            if seen.intersection(keys):
                continue
            seen.update(keys)
            results.append({**row, "rank": len(results) + 1})
            if len(results) == POOL_LIMIT:
                break
        if self.identity(request) != identity:
            raise CollectionConflict("Search scope changed; run the search again")
        if any(signature(Path(path)) != expected for path, expected in head_files.items()):
            raise CollectionConflict("Search source changed; run the search again")
        deferred = head.pop("_record_payload", None)
        payload = {
            **head,
            "results": results,
            "pagination": {
                "policy": "preserved-head-deep-tail-v1",
                "head_count": len(head["results"]),
                "tail_ranking": (
                    "hybrid"
                    if request.get(
                        "mode", "visual" if request.get("kind") == "similar" else "hybrid"
                    )
                    == "hybrid"
                    else "native"
                ),
            },
        }
        if deferred is not None:
            # Head candidates retain their original feature/budget provenance.
            # Tail labels support query memory/export, but lack core training features.
            with FeedbackStore(self.root) as store:
                run = deferred
                for key in ("run_id", "created_at"):
                    run.pop(key, None)
                run["results"] = results
                run["ranking"] = {
                    **run["ranking"],
                    "pagination": {
                        "policy": "preserved-head-deep-tail-v1",
                        "pool_limit": POOL_LIMIT,
                        "head_count": len(head["results"]),
                        "tail_ranking": (
                            "hybrid"
                            if request.get(
                                "mode", "visual" if request.get("kind") == "similar" else "hybrid"
                            )
                            == "hybrid"
                            else "native"
                        ),
                    },
                }
                payload["run_id"] = store.add_run(run)
        files = {}
        for row in results:
            for path in row["paths"]:
                try:
                    files[path] = signature(Path(path))
                except OSError as error:
                    raise CollectionConflict(
                        "Search source changed; run the search again"
                    ) from error
        key = identifier()
        snapshot = {
            "request": dict(request),
            "identity": identity,
            "payload": payload,
            "files": files,
            "created": monotonic(),
            "size": request["limit"],
            "truncated": len(results) == POOL_LIMIT,
        }
        with self.lock:
            self.snapshots[key] = snapshot
            self._prune()
            return self._page(key, snapshot, 0)

    def _prune(self):
        for key in list(self.snapshots):
            if monotonic() - self.snapshots[key]["created"] > TTL_SECONDS:
                del self.snapshots[key]
        while len(self.snapshots) > MAX_SNAPSHOTS:
            self.snapshots.popitem(last=False)

    def page(self, cursor: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}:[0-9]{1,4}", cursor):
            raise ValueError("Invalid search cursor")
        key, raw = cursor.split(":")
        offset = int(raw)
        with self.lock:
            self._prune()
            snapshot = self.snapshots.get(key)
            if snapshot is None:
                raise CollectionConflict("Search expired; run the search again")
            if not 0 < offset < len(snapshot["payload"]["results"]) or offset % snapshot["size"]:
                raise ValueError("Invalid search page")
            if self.identity(snapshot["request"]) != snapshot["identity"]:
                raise CollectionConflict("Search scope changed; run the search again")
            for path, expected in snapshot["files"].items():
                try:
                    current = signature(Path(path))
                except OSError as error:
                    raise CollectionConflict(
                        "Search source changed; run the search again"
                    ) from error
                if current != expected:
                    raise CollectionConflict("Search source changed; run the search again")
            self.snapshots.move_to_end(key)
            return self._page(key, snapshot, offset)

    def _page(self, key: str, snapshot: dict, offset: int) -> dict:
        payload = snapshot["payload"]
        end = offset + snapshot["size"]
        total = len(payload["results"])
        page_rows = payload["results"][offset:end]
        request = snapshot["request"]
        if (
            request.get("query")
            and request.get("use_feedback", True)
            and (self.root / "feedback.sqlite3").is_file()
        ):
            catalog = Catalog(self.root, readonly=True)
            try:
                collection_id = (
                    "library-v1"
                    if request.get("all_folders")
                    else catalog.snapshot(request["collection"])[0]["id"]
                )
            finally:
                catalog.close()
            with FeedbackStore(self.root, readonly=True) as store:
                decisions = store.memory(
                    collection_id,
                    request["query"],
                    memory_scope(
                        request.get("virtual_collection", "") or "", payload.get("interpretation")
                    ),
                )
            aliases = {row["image_id"]: row for row in decisions}
            refreshed = []
            for row in page_rows:
                matches = [
                    aliases[key]
                    for key in [row["image_id"], *row.get("duplicate_ids", [])]
                    if key in aliases
                ]
                decision = (
                    max(matches, key=lambda value: value["event_id"])
                    if matches
                    else row.get("feedback")
                )
                if not decision or decision.get("label") != "irrelevant":
                    refreshed.append({**row, "feedback": decision})
            page_rows = refreshed
        return deepcopy(
            {
                **payload,
                "results": page_rows,
                "next_cursor": f"{key}:{end}" if end < total else None,
                "total": total,
                "pool_limit": POOL_LIMIT,
                "truncated": snapshot["truncated"],
                "offset": offset,
            }
        )
