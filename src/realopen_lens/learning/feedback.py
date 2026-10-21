"""Opt-in production search recording and explicit, auditable relevance labels."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from realopen_lens.models.manager import digest
from realopen_lens.retrieval import images, search, ranking, identifiers
from realopen_lens.retrieval import candidates as candidate_policy
from realopen_lens.learning import features, logistic
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog, default_directory
from realopen_lens.storage.feedback import FeedbackStore


@dataclass(frozen=True)
class RecordedSearch:
    run_id: str
    results: list[search.SearchResult]


def _record(
    operation: Callable[[], list[search.SearchResult]],
    query: dict,
    options: dict,
    trace: dict,
    index_dir: Path | None,
) -> RecordedSearch:
    root = index_dir or default_directory()
    collection_name = options.get("collection", "personal")
    catalog = Catalog(root, readonly=True)
    try:
        collection, generation = catalog.snapshot(collection_name)
        manifest = load_manifest(catalog, generation)
        scope = None
        if options.get("virtual_collection") is not None:
            from realopen_lens.organization.collections import resolve_scope, ensure_scope

            _, scope = resolve_scope(
                root,
                collection["id"],
                options["virtual_collection"],
                options.get("virtual_revision"),
            )
            options["virtual_revision"] = scope["revision"]
        results = operation()
        if scope is not None:
            ensure_scope(root, collection["id"], scope)
        _, current = catalog.snapshot(collection_name)
        if current["id"] != generation["id"] or (
            trace.get("generation", generation["id"]) != generation["id"]
        ):
            raise ValueError("Collection changed while recording search; retry")
    finally:
        catalog.close()
    if query["kind"] == "image":
        query = {"kind": "image", "image_id": trace["query_image_id"]}
    config = {
        "mode": options.get("mode", "visual" if query["kind"] == "image" else "hybrid"),
        "limit": options.get("limit", 10),
        "threads": options.get("threads", 2),
        "siglip_precision": manifest["config"]["spec"].get("siglip_precision", "int8"),
        "candidate_depths": trace.get("candidate_depths"),
        "reranker_sha256": trace.get("reranker_sha256"),
        "reranker_scope": trace.get("reranker_scope"),
        "query_runtime": "resident" if options.get("query_encoder") is not None else "one-shot",
    }
    if query["kind"] == "text":
        config["feedback_memory"] = options.get("feedback_trace", {})
    if query["kind"] == "image":
        config.update(
            include_self=options.get("include_self", False),
            reused_embedding=trace["reused_embedding"],
        )
    if config["mode"] == "lexical":
        config["query_runtime"] = "none"
    elif query["kind"] == "image" and trace["reused_embedding"]:
        config["query_runtime"] = "stored"
    from realopen_lens.organization import duplicates, fingerprints
    from realopen_lens.retrieval import memory

    duplicate_snapshot = duplicates.snapshot(root, manifest)
    if duplicate_snapshot is not None:
        config["duplicate_snapshot"] = duplicate_snapshot["key"]
        config["duplicate_policy"] = duplicate_snapshot["policy"]
    returned_ids = {r.image_id for r in results}
    candidates = {
        r["image_id"]: r for r in trace.get("candidates", []) if r["image_id"] in returned_ids
    }
    source_hashes = {}
    for module in (
        search,
        images,
        ranking,
        identifiers,
        candidate_policy,
        features,
        logistic,
        duplicates,
        fingerprints,
        memory,
    ):
        if module.__file__ is None:
            raise ValueError("Ranking source provenance is unavailable")
        source = Path(module.__file__)
        source_hashes[source.name] = digest(source)
    payload = {
        "version": 1,
        "query": query,
        "collection": collection_name,
        "collection_id": collection["id"],
        "generation": generation["id"],
        "manifest_sha256": generation["checksum"],
        "pipeline": manifest["pipeline"],
        "encoder_spec": manifest["config"]["spec"],
        "ranking": config,
        "ranking_sources": source_hashes,
        "candidate_schema": 1,
        "results": [asdict(r) for r in results],
        "candidates": candidates,
    }
    if scope is not None:
        payload["scope"] = scope
    with FeedbackStore(root) as store:
        run_id = store.add_run(payload)
    return RecordedSearch(run_id, results)


def search_and_record(
    query: str, *, index_dir: Path | None = None, **options: Any
) -> RecordedSearch:
    """Execute production text search, then save its provenance and returned results.

    Ordinary searches are never recorded automatically. An optional resident
    query_encoder can be supplied; trace/profile outputs remain owned here.
    """
    if "trace" in options or "profile" in options:
        raise ValueError("Recorded search owns its trace and profile")
    trace: dict = {}
    options.setdefault("feedback_trace", {})
    kwargs: dict[str, Any] = {"trace": trace} if options.get("mode", "hybrid") == "hybrid" else {}
    return _record(
        lambda: search.search_collection(query, index_dir=index_dir, **kwargs, **options),
        {"kind": "text", "text": query},
        options,
        trace,
        index_dir,
    )


def similar_and_record(
    image: Path | None = None, *, index_dir: Path | None = None, **options: Any
) -> RecordedSearch:
    """Record production image similarity search without storing the external path."""
    if "trace" in options:
        raise ValueError("Recorded search owns its trace")
    trace: dict = {}
    return _record(
        lambda: images.search_similar(image, index_dir=index_dir, trace=trace, **options),
        {"kind": "image"},
        options,
        trace,
        index_dir,
    )


def saved_runs(*, index_dir: Path | None = None, limit: int = 20) -> list[dict]:
    root = index_dir or default_directory()
    if not 1 <= limit <= 100:
        raise ValueError("Use 1–100 saved runs")
    if not (root / "feedback.sqlite3").exists():
        return []
    with FeedbackStore(root, readonly=True) as store:
        return store.runs(limit)


def saved_search(run_id: str, *, index_dir: Path | None = None) -> dict:
    with FeedbackStore(index_dir or default_directory(), readonly=True) as store:
        return {**store.run(run_id), "judgments": store.judgments(run_id)}


def change_judgment(
    run_id: str,
    image_id: str,
    label: str | None,
    *,
    index_dir: Path | None = None,
    expected_event_id: int | None = None,
) -> dict:
    from realopen_lens.retrieval.memory import aliases_for_run

    if expected_event_id is not None and expected_event_id < 0:
        raise ValueError("Expected event ID must be nonnegative")
    root = index_dir or default_directory()
    if not (root / "feedback.sqlite3").is_file():
        raise ValueError("No saved searches; use --save-run first")
    with FeedbackStore(root) as store:
        run = store.run(run_id)
        aliases = aliases_for_run(root, run, image_id)
        judgment, decision = store.change(
            run_id, image_id, label, aliases=aliases, expected_event_id=expected_event_id
        )
        return {"judgment": judgment, "memory": decision}


def judge(run_id: str, image_id: str, label: str, *, index_dir: Path | None = None) -> dict | None:
    if label not in {"relevant", "irrelevant"}:
        raise ValueError("Label must be relevant or irrelevant")
    return change_judgment(run_id, image_id, label, index_dir=index_dir)["judgment"]


def remove_judgment(run_id: str, image_id: str, *, index_dir: Path | None = None) -> None:
    change_judgment(run_id, image_id, None, index_dir=index_dir)


def judgment_history(run_id: str, *, index_dir: Path | None = None) -> list[dict]:
    with FeedbackStore(index_dir or default_directory(), readonly=True) as store:
        return store.history(run_id)


def superseded_judgments(store: FeedbackStore, root: Path, run: dict) -> set[str]:
    """Find contradicted or cleared historical labels without deleting history."""
    from realopen_lens.storage.feedback import memory_key
    from realopen_lens.retrieval.memory import aliases_for_run

    key = memory_key(run)
    if key is None:
        return set()
    rows = store.memory(key[0], run["query"]["text"], key[2])
    invalid = set()
    for judgment in store.judgments(run["run_id"]):
        aliases = aliases_for_run(root, run, judgment["image_id"])
        latest = max(
            (row for row in rows if row["image_id"] in aliases),
            key=lambda row: row["event_id"],
            default=None,
        )
        if latest and latest["label"] != judgment["label"]:
            invalid.add(judgment["image_id"])
    return invalid


def export_feedback(output: Path, *, index_dir: Path | None = None, overwrite: bool = False) -> int:
    """Atomically export active explicit labels from one consistent SQLite snapshot.

    Unjudged/removed results never become negatives. No split assignments, implicit
    labels, or training are performed. Paths and OCR previews are excluded.
    """
    root = index_dir or default_directory()
    output = output.expanduser().resolve()
    if output in {
        (root / name).resolve()
        for name in (
            "catalog.sqlite3",
            "feedback.sqlite3",
            "catalog.sqlite3-wal",
            "catalog.sqlite3-shm",
            "feedback.sqlite3-wal",
            "feedback.sqlite3-shm",
            "organization.sqlite3",
            "organization.sqlite3-wal",
            "organization.sqlite3-shm",
        )
    }:
        raise ValueError("Export cannot replace an index database")
    if output.is_relative_to((root / "duplicates").resolve()):
        raise ValueError("Export cannot replace duplicate fingerprints or reviews")
    if output.exists() and not overwrite:
        raise ValueError("Export already exists; use overwrite explicitly")
    temporary = output.with_name(output.name + ".tmp-" + uuid4().hex)
    count = 0
    obsolete: dict[str, set[str]] = {}
    try:
        with FeedbackStore(root, readonly=True) as store:
            store.db.execute("BEGIN")
            cursor = store.db.execute(
                "SELECT j.*,s.payload,s.created_at FROM judgments j "
                "JOIN searches s ON s.id=j.run_id "
                "ORDER BY s.rowid,j.image_id"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for row in cursor:
                    run = json.loads(row["payload"])
                    if row["run_id"] not in obsolete:
                        obsolete[row["run_id"]] = superseded_judgments(
                            store, root, {**run, "run_id": row["run_id"]}
                        )
                    result = next(r for r in run["results"] if r["image_id"] == row["image_id"])
                    candidate = run["candidates"].get(row["image_id"])
                    record = {
                        "version": 1,
                        "run_id": row["run_id"],
                        "query": run["query"],
                        "image_id": row["image_id"],
                        "label": int(row["label"] == "relevant"),
                        "judgment": row["label"],
                        "revision": row["revision"],
                        "judged_at": row["updated_at"],
                        "searched_at": row["created_at"],
                        **{
                            key: run[key]
                            for key in (
                                "collection",
                                "collection_id",
                                "generation",
                                "manifest_sha256",
                                "pipeline",
                                "encoder_spec",
                                "ranking",
                                "ranking_sources",
                            )
                        },
                        "result": {
                            key: result[key]
                            for key in ("rank", "visual_similarity", "width", "height", "status")
                        },
                        "candidate": candidate,
                        "candidate_schema": run.get("candidate_schema") if candidate else None,
                    }
                    if row["image_id"] in obsolete[row["run_id"]]:
                        record["superseded"] = True
                    if result.get("duplicate_ids"):
                        record["duplicate_ids"] = result["duplicate_ids"]
                    if "scope" in run:
                        record["scope"] = run["scope"]
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
                stream.flush()
                os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as error:
                raise ValueError("Export already exists; use overwrite explicitly") from error
    finally:
        temporary.unlink(missing_ok=True)
    return count
