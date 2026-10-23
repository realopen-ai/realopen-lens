"""Exact-query relevance memory, separate from model scores and evaluation."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from realopen_lens.organization.duplicates import snapshot
from realopen_lens.storage.feedback import FeedbackStore, MEMORY_POLICY


def _pixels(root: Path, manifest: dict) -> dict[str, str]:
    data = snapshot(root, manifest)
    return data.get("pixels", {}) if data else {}


def _content(root: Path, key: str, pixels: dict[str, str]) -> str:
    if key in pixels:
        return "pixels:" + pixels[key]
    if pixels:
        from realopen_lens.organization.duplicates import _cached

        old = _cached(root / "duplicates/cache" / (key + ".json"), key)
        if old:
            return "pixels:" + old["pixels"]
    return "bytes:" + key


def resolve(
    root: Path,
    collection_id: str,
    query: str,
    scope_key: str,
    manifest: dict,
    available: dict,
    *,
    content_pixels: dict[str, str] | None = None,
) -> tuple[dict[str, dict], dict]:
    summary = {
        "enabled": True,
        "policy": MEMORY_POLICY,
        "applied": False,
        "positive_groups": 0,
        "hidden_groups": 0,
        "max_event_id": 0,
    }
    if not (root / "feedback.sqlite3").is_file():
        return {}, summary
    with FeedbackStore(root, readonly=True) as store:
        store.db.execute("BEGIN")
        rows = store.memory(collection_id, query, scope_key)
    pixels = (
        (content_pixels if content_pixels is not None else _pixels(root, manifest)) if rows else {}
    )
    content = {key: _content(root, key, pixels) for key in available}
    latest: dict[str, dict] = {}
    for row in rows:
        group = _content(root, row["image_id"], pixels)
        if row["event_id"] > latest.get(group, {}).get("event_id", 0):
            latest[group] = row
    active = {group: latest[group] for group in set(content.values()) if group in latest}
    decisions = {key: active[group] for key, group in content.items() if group in active}
    positive = sum(row["label"] == "relevant" for row in active.values())
    negative = sum(row["label"] == "irrelevant" for row in active.values())
    summary.update(
        positive_groups=positive,
        hidden_groups=negative,
        applied=bool(positive or negative),
        max_event_id=max((row["event_id"] for row in rows), default=0),
        snapshot_sha256=hashlib.sha256(
            json.dumps({"decisions": rows, "content_groups": content}, sort_keys=True).encode()
        ).hexdigest(),
    )
    return decisions, summary


def apply(ranking: list[str], decisions: dict[str, dict]) -> list[str]:
    positive = [key for key in ranking if decisions.get(key, {}).get("label") == "relevant"]
    seen = set(ranking)
    positive.extend(
        sorted(
            (
                key
                for key, value in decisions.items()
                if value["label"] == "relevant" and key not in seen
            ),
            key=lambda key: (-decisions[key]["event_id"], key),
        )
    )
    unknown = [
        key
        for key in ranking
        if decisions.get(key, {}).get("label") not in {"relevant", "irrelevant"}
    ]
    return positive + unknown


def aliases_for_run(root: Path, run: dict, image_id: str) -> tuple[str, ...]:
    row: dict = next((r for r in run["results"] if r["image_id"] == image_id), {})
    aliases = {image_id, *row.get("duplicate_ids", [])}
    if run.get("library_sources") is not None:
        from realopen_lens.organization.duplicates import _cached

        # The current run may contain only a new format alias after an old
        # source disappeared. Include historical aliases before the optimistic
        # event check, just as collection-scoped query memory does.
        pixels = {}
        for key in aliases:
            cached = _cached(root / "duplicates/cache" / (key + ".json"), key)
            if cached:
                pixels[key] = cached["pixels"]
        if pixels and run["query"]["kind"] == "text" and (root / "feedback.sqlite3").is_file():
            if len(set(pixels.values())) != 1:
                raise ValueError("Library duplicate identities disagree; scan duplicates again")
            pixels[image_id] = next(iter(pixels.values()))
            group = _content(root, image_id, pixels)
            with FeedbackStore(root, readonly=True) as store:
                rows = store.memory(
                    run["collection_id"],
                    run["query"]["text"],
                    (run.get("scope") or {}).get("id", ""),
                )
            aliases.update(
                row["image_id"] for row in rows if _content(root, row["image_id"], pixels) == group
            )
        return tuple(sorted(aliases))
    if (root / "catalog.sqlite3").is_file():
        from realopen_lens.storage.catalog import Catalog
        from realopen_lens.storage.artifacts import load_manifest

        catalog = Catalog(root, readonly=True)
        try:
            if (
                catalog.db.execute(
                    "SELECT 1 FROM collections WHERE id=?", (run["collection_id"],)
                ).fetchone()
                is None
            ):
                return tuple(sorted(aliases))  # Historical feedback outlives unregistration.
            found, generation = catalog.snapshot(catalog.collection_name(run["collection_id"]))
            if found["id"] == run["collection_id"]:
                pixels = _pixels(root, load_manifest(catalog, generation))
                group = _content(root, image_id, pixels)
                candidates = set(pixels)
                if (root / "feedback.sqlite3").is_file() and run["query"]["kind"] == "text":
                    with FeedbackStore(root, readonly=True) as store:
                        candidates.update(
                            item["image_id"]
                            for item in store.memory(
                                run["collection_id"],
                                run["query"]["text"],
                                (run.get("scope") or {}).get("id", ""),
                            )
                        )
                aliases.update(key for key in candidates if _content(root, key, pixels) == group)
        finally:
            catalog.close()
    return tuple(sorted(aliases))


def hidden(
    root: Path,
    query: str,
    *,
    collection: str = "personal",
    virtual_collection: str | None = None,
    virtual_revision: int | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Browse available hidden exact groups without encoder inference."""
    from realopen_lens.storage.catalog import Catalog
    from realopen_lens.storage.artifacts import load_manifest
    from realopen_lens.retrieval.search import source_paths, search_results
    from realopen_lens.organization.duplicates import exact_groups, collapse
    from realopen_lens.organization.collections import resolve_scope, ensure_scope

    if not query.strip() or len(query) > 4096 or offset < 0 or not 1 <= limit <= 100:
        raise ValueError("Use a valid query and page range")
    catalog = Catalog(root, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
        paths = source_paths(Path(found["root"]), manifest)
        scope = None
        if virtual_collection is not None:
            allowed, scope = resolve_scope(root, found["id"], virtual_collection, virtual_revision)
            paths = {key: value for key, value in paths.items() if key in allowed}
        elif virtual_revision is not None:
            raise ValueError("Virtual revision requires a virtual collection")
        decisions, summary = resolve(
            root, found["id"], query, virtual_collection or "", manifest, paths
        )
        ranking = sorted(
            (key for key in decisions if decisions[key]["label"] == "irrelevant"),
            key=lambda key: (-decisions[key]["event_id"], key),
        )
        ranking = collapse(ranking, exact_groups(root, manifest, paths))
        rows = search_results(
            catalog, manifest, paths, ranking[offset : offset + limit], {}, deduplicate=True
        )
        results = [asdict(replace(row, feedback=decisions[row.image_id])) for row in rows]
        if scope:
            ensure_scope(root, found["id"], scope)
        return {
            "results": results,
            "total": len(ranking),
            "offset": offset,
            "generation": generation["id"],
            "feedback": summary,
        }
    finally:
        catalog.close()
