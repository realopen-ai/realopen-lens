"""Preview and publish a new workspace from compatible, read-only source snapshots."""

from collections import defaultdict
from contextlib import closing, ExitStack
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4

from realopen_lens.models.manager import digest
from realopen_lens.organization.duplicates import snapshot
from realopen_lens.runtime.jobs import writer_lock
from realopen_lens.storage.artifacts import atomic_bytes, load_manifest
from realopen_lens.storage.catalog import Catalog, local_path, now
from realopen_lens.storage.feedback import FeedbackStore, memory_key
from realopen_lens.storage.organization import OrganizationStore


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def checksum(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def rows(db, table):
    return [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def _read(root: Path, progress) -> dict:
    progress({"phase": "inspecting", "source": str(root)})
    catalog = Catalog(root, readonly=True)
    try:
        catalog.db.execute("BEGIN")
        collections = rows(catalog.db, "collections")
        if len(collections) != 1:
            raise ValueError("Workspace import currently requires one source collection per index")
        found = collections[0]
        row = catalog.db.execute(
            "SELECT * FROM generations WHERE id=?", (found["active_generation"],)
        ).fetchone()
        if row is None:
            raise ValueError("Source collection has no published index")
        active = dict(row)
        if catalog.db.execute(
            "SELECT 1 FROM jobs WHERE status IN ('running','queued','pending')"
        ).fetchone():
            raise ValueError("Finish source indexing before importing its workspace")
        manifest = load_manifest(catalog, active)
        identity = {key: found[key] for key in ("id", "name", "root", "active_generation")}
        identity["manifest_sha256"] = active["checksum"]
        catalog_hash = checksum(list(catalog.db.iterdump()))
        files = {}

        def include(relative, expected=None):
            path = local_path(root, relative)
            if not path.is_file():
                raise ValueError(f"Required import artifact is missing: {path}")
            actual = digest(path)
            if expected is not None and expected != actual:
                raise ValueError(f"Import artifact checksum differs: {path}")
            files[relative] = {"sha256": actual, "bytes": path.stat().st_size}
            if len(files) % 25 == 0:
                progress({"phase": "checking_artifacts", "files": len(files), "source": str(root)})

        include(active["manifest"], active["checksum"])
        for item in manifest["artifacts"].values():
            include(item["path"], item["sha256"])
        for item in manifest["images"]:
            include(item["thumbnail"], item["thumbnail_sha256"])
        # Keep reusable successful stage caches; unavailable historical caches are recomputed.
        for item in catalog.db.execute(
            "SELECT artifact,checksum FROM analyses WHERE artifact IS NOT NULL"
        ):
            path = local_path(root, item["artifact"])
            if path.is_file() and digest(path) == item["checksum"]:
                include(item["artifact"], item["checksum"])
        # Historical generations remain usable when their complete artifacts are present.
        for generation in rows(catalog.db, "generations"):
            if generation["id"] == active["id"]:
                continue
            try:
                old = load_manifest(catalog, generation)
            except (ValueError, OSError):
                continue
            include(generation["manifest"], generation["checksum"])
            for item in old["artifacts"].values():
                include(item["path"], item["sha256"])
        grouped = snapshot(root, manifest)
    finally:
        catalog.close()
    feedback: dict[str, list[dict]] = {"searches": [], "events": [], "judgments": []}
    if (root / "feedback.sqlite3").is_file():
        with FeedbackStore(root, readonly=True) as store:
            store.db.execute("BEGIN")
            feedback = {key: rows(store.db, key) for key in feedback}
            for run in feedback["searches"]:
                if run["collection_id"] != found["id"]:
                    raise ValueError(
                        "Feedback contains another source collection; import separately"
                    )
    organization: dict[str, list[dict]] = {"virtual_collections": [], "members": []}
    if (root / "organization.sqlite3").is_file():
        with OrganizationStore(root, readonly=True) as store:
            store.db.execute("BEGIN")
            organization = {key: rows(store.db, key) for key in organization}
            if any(
                row["collection_id"] != found["id"] for row in organization["virtual_collections"]
            ):
                raise ValueError("Virtual collections belong to another source")
    reviews = {}
    review_path = root / "duplicates/reviews" / (found["id"] + ".json")
    if review_path.is_file():
        if grouped is None:
            raise ValueError("Duplicate reviews require their matching fingerprint snapshot")
        reviews = json.loads(review_path.read_text(encoding="utf-8"))
        if any(value not in {"confirmed", "dismissed", "unreviewed"} for value in reviews.values()):
            raise ValueError("Invalid duplicate-review state")
    fingerprints = {}
    cached_pixels = {}
    if grouped:
        from realopen_lens.organization.duplicates import _cached

        for path in sorted((root / "duplicates/cache").glob("*.json")):
            cached = _cached(path, path.stem)
            if cached is None:
                raise ValueError("Invalid duplicate fingerprint cache")
            fingerprints[path.name] = digest(path)
            cached_pixels[path.stem] = cached["pixels"]
    return {
        "fingerprints": fingerprints,
        "cached_pixels": cached_pixels,
        "root": str(root),
        "identity": identity,
        "catalog_sha256": catalog_hash,
        "files": files,
        "feedback": feedback,
        "organization": organization,
        "duplicates": grouped,
        "reviews": reviews,
        "images": len(manifest["images"]),
    }


def _merge(sources: list[dict]) -> dict:
    runs: dict = {}
    events: dict = {}
    edges: dict = defaultdict(set)
    indegree: dict = defaultdict(int)
    decisions = defaultdict(list)
    virtuals: dict = {}
    memberships: dict = {}
    names: dict = {}
    reviews: dict = {}
    grouped = None
    pixels: dict[str, str] = {}
    for source in sources:
        if source["identity"] != sources[0]["identity"]:
            raise ValueError("Source collection identity, root or active generation differs")
        if source["duplicates"]:
            if grouped is not None and any(
                grouped.get(key) != source["duplicates"].get(key)
                for key in ("key", "policy", "pixels", "exact", "near")
            ):
                raise ValueError("Duplicate fingerprint snapshots conflict")
            grouped = source["duplicates"]
            for key, value in {**source["cached_pixels"], **grouped["pixels"]}.items():
                if key in pixels and pixels[key] != value:
                    raise ValueError("Pixel identity caches conflict")
                pixels[key] = value
    for source in sources:
        saved = {}
        for row in source["feedback"]["searches"]:
            value = {**row, "payload": json.loads(row["payload"])}
            if row["id"] in runs and runs[row["id"]] != value:
                raise ValueError(f"Saved search conflicts: {row['id']}")
            runs[row["id"]] = saved[row["id"]] = value
        latest, previous, nodes = {}, None, set()
        latest_pairs = {}
        for event in source["feedback"]["events"]:
            node = (event["run_id"], event["image_id"], event["revision"])
            value = {key: val for key, val in event.items() if key != "id"}
            if node in events and events[node] != value:
                raise ValueError("Feedback event histories conflict")
            events[node] = value
            nodes.add(node)
            indegree.setdefault(node, 0)
            if previous is not None and node not in edges[previous]:
                edges[previous].add(node)
                indegree[node] += 1
            previous = node
            latest_pairs[node[:2]] = value
            run = saved[event["run_id"]]["payload"]
            key = memory_key(run)
            if key is not None:
                image = (
                    "pixels:" + pixels[event["image_id"]]
                    if event["image_id"] in pixels
                    else event["image_id"]
                )
                latest[(*key, image)] = node
        judgments = {(r["run_id"], r["image_id"]): r for r in source["feedback"]["judgments"]}
        expected = {key: value for key, value in latest_pairs.items() if value["label"] is not None}
        if set(judgments) != set(expected) or any(
            judgments[key]["label"] != value["label"]
            or judgments[key]["revision"] != value["revision"]
            for key, value in expected.items()
        ):
            raise ValueError("Feedback judgments do not match event history")
        for decision_key, node in latest.items():
            decisions[decision_key].append((node, nodes))
        for row in source["organization"]["virtual_collections"]:
            key = row["id"]
            members = [r for r in source["organization"]["members"] if r["virtual_id"] == key]
            if key in virtuals and (virtuals[key] != row or memberships[key] != members):
                raise ValueError(
                    f"Virtual collection revisions or membership conflict: {row['name']}"
                )
            if row["name_key"] in names and names[row["name_key"]] != key:
                raise ValueError("Different virtual collections have the same name")
            virtuals[key], memberships[key], names[row["name_key"]] = row, members, key
        for key, value in source["reviews"].items():
            if key in reviews and reviews[key] != value:
                raise ValueError(f"Duplicate reviews conflict: {key}")
            reviews[key] = value
    for values in decisions.values():
        for i, (node, history) in enumerate(values):
            for other, other_history in values[i + 1 :]:
                if (
                    events[node]["label"] != events[other]["label"]
                    and node not in other_history
                    and other not in history
                ):
                    raise ValueError(
                        "Independent query judgments conflict; choose an authoritative source"
                    )
    ready = [(events[node]["created_at"], node) for node in events if indegree[node] == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        _, node = heapq.heappop(ready)
        ordered.append(events[node])
        for following in edges[node]:
            indegree[following] -= 1
            if indegree[following] == 0:
                heapq.heappush(ready, (events[following]["created_at"], following))
    if len(ordered) != len(events):
        raise ValueError("Source feedback histories have incompatible ordering")
    return {
        "runs": runs,
        "events": ordered,
        "virtuals": virtuals,
        "memberships": memberships,
        "duplicates": grouped,
        "reviews": reviews,
    }


def destination_mode(target: Path) -> str:
    if not target.exists():
        return "new"
    operational = {"api-server", "api-tasks", "api-uploads", "indexing.lock"}
    if not target.is_dir() or any(path.name not in operational for path in target.iterdir()):
        raise ValueError("Use a new destination or an uninitialized workspace without saved data")
    return "uninitialized"


def source_roots(sources: list[Path], target: Path) -> list[Path]:
    roots = [path.expanduser().resolve() for path in sources]
    if not 1 <= len(roots) <= 16 or len(set(roots)) != len(roots):
        raise ValueError("Supply 1–16 distinct source indexes")
    if any(target.is_relative_to(root) or root.is_relative_to(target) for root in roots):
        raise ValueError("Use a new destination outside all source indexes")
    destination_mode(target)
    return roots


def preview(sources: list[Path], destination: Path, on_progress=None) -> dict:
    progress = on_progress or (lambda value: None)
    target = destination.expanduser().resolve()
    roots = source_roots(sources, target)
    mode = destination_mode(target)
    states = [_read(root, progress) for root in roots]
    merged = _merge(states)
    summary = {
        "version": 1,
        "destination": str(target),
        "destination_mode": mode,
        "sources": [str(root) for root in roots],
        "collection": states[0]["identity"],
        "images": states[0]["images"],
        "searches": len(merged["runs"]),
        "events": len(merged["events"]),
        "judgments": sum(
            value["label"] is not None
            for value in {
                (row["run_id"], row["image_id"]): row for row in merged["events"]
            }.values()
        ),
        "virtual_collections": len(merged["virtuals"]),
        "duplicate_reviews": len(merged["reviews"]),
        "copy_bytes": sum(item["bytes"] for item in states[0]["files"].values()),
        "exclusions": [
            "model weights",
            "reranker artifacts and activation",
            "API task history",
            "original images",
        ],
        "source_sha256": [checksum(state) for state in states],
    }
    summary["plan_sha256"] = checksum(summary)
    return summary


def initialize(
    sources: list[Path],
    destination: Path,
    expected_plan: str,
    *,
    models: Path,
    on_progress=None,
    owned_server: Path | None = None,
) -> dict:
    progress = on_progress or (lambda value: None)
    target = destination.expanduser().resolve()
    source_roots(sources, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(target.parent / ("." + target.name + "-import-lock")), ExitStack() as guards:
        if target.exists():
            try:
                if owned_server is None or owned_server.resolve() != target:
                    guards.enter_context(writer_lock(target / "api-server"))
                guards.enter_context(writer_lock(target))
            except ValueError as error:
                raise ValueError(
                    "Destination is in use; stop its server and indexing before importing"
                ) from error
        plan = preview(sources, target, progress)
        if plan["plan_sha256"] != expected_plan:
            raise ValueError("Import preview changed; inspect a fresh preview before applying")
        states = [_read(Path(root), progress) for root in plan["sources"]]
        if [checksum(state) for state in states] != plan["source_sha256"]:
            raise ValueError("Source changed after import preview")
        merged = _merge(states)
        stage = target.with_name("." + target.name + ".import-" + uuid4().hex)
        stage.mkdir()
        try:
            progress({"phase": "copying_catalog"})
            with closing(
                sqlite3.connect(
                    Path(plan["sources"][0], "catalog.sqlite3").as_uri() + "?mode=ro", uri=True
                )
            ) as source:
                source.execute("BEGIN")
                if checksum(list(source.iterdump())) != states[0]["catalog_sha256"]:
                    raise ValueError("Source catalog changed")
                with closing(sqlite3.connect(stage / "catalog.sqlite3")) as output_db:
                    source.backup(output_db)
            total = len(states[0]["files"])
            for position, (relative, metadata) in enumerate(states[0]["files"].items(), 1):
                progress({"phase": "copying_artifacts", "step": position - 1, "max_steps": total})
                output = local_path(stage, relative)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local_path(Path(plan["sources"][0]), relative), output)
                if digest(output) != metadata["sha256"]:
                    raise ValueError("Source artifact changed during copying")
            progress({"phase": "importing_feedback"})
            with FeedbackStore(stage) as store:
                with store.db:
                    for key, run in merged["runs"].items():
                        store.db.execute(
                            "INSERT INTO searches VALUES (?,?,?,?)",
                            (
                                key,
                                run["created_at"],
                                run["collection_id"],
                                canonical(run["payload"]).decode(),
                            ),
                        )
                        store._index_run(key, run["payload"])
                    for row in merged["events"]:
                        store.db.execute(
                            "INSERT INTO events(run_id,image_id,label,revision,created_at) "
                            "VALUES (?,?,?,?,?)",
                            tuple(
                                row[key]
                                for key in ("run_id", "image_id", "label", "revision", "created_at")
                            ),
                        )
                    store.db.execute(
                        "INSERT OR REPLACE INTO query_memory "
                        "SELECT m.collection_id,m.query_key,m.scope_key,"
                        "e.image_id,e.label,e.id,e.run_id "
                        "FROM events e JOIN memory_runs m ON m.run_id=e.run_id ORDER BY e.id"
                    )
                    current = {(row["run_id"], row["image_id"]): row for row in merged["events"]}
                    for row in current.values():
                        if row["label"] is not None:
                            store.db.execute(
                                "INSERT INTO judgments VALUES (?,?,?,?,?)",
                                tuple(
                                    row[key]
                                    for key in (
                                        "run_id",
                                        "image_id",
                                        "label",
                                        "revision",
                                        "created_at",
                                    )
                                ),
                            )
                store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if merged["virtuals"]:
                with OrganizationStore(stage) as store:
                    with store.db:
                        for row in merged["virtuals"].values():
                            store.db.execute(
                                "INSERT INTO virtual_collections VALUES (?,?,?,?,?,?,?)",
                                tuple(row.values()),
                            )
                        for members in merged["memberships"].values():
                            for row in members:
                                store.db.execute(
                                    "INSERT INTO members VALUES (?,?,?)", tuple(row.values())
                                )
                    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if merged["duplicates"] is not None:
                grouped = merged["duplicates"]
                atomic_bytes(stage / "duplicates" / (grouped["key"] + ".json"), canonical(grouped))
                atomic_bytes(
                    stage / "duplicates/reviews" / (plan["collection"]["id"] + ".json"),
                    canonical(merged["reviews"]),
                )
                # Cache fingerprints also preserve retired exact-image aliases.
                for root in map(Path, plan["sources"]):
                    state = next(item for item in states if item["root"] == str(root))
                    for name, expected in state["fingerprints"].items():
                        path = root / "duplicates/cache" / name
                        if digest(path) != expected:
                            raise ValueError("Duplicate fingerprint changed during import")
                        output = stage / "duplicates/cache" / path.name
                        if output.exists() and json.loads(output.read_bytes()) != json.loads(
                            path.read_bytes()
                        ):
                            raise ValueError("Duplicate fingerprint caches conflict")
                        output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, output)
            progress({"phase": "verifying_sources"})
            if preview(sources, target, progress)["plan_sha256"] != expected_plan:
                raise ValueError("Source changed during import; no workspace published")
            metadata = {
                "version": 1,
                "id": uuid4().hex,
                "created_at": now(),
                "models_directory": str(models.expanduser().resolve()),
                "import": plan,
            }
            atomic_bytes(stage / "workspace.json", canonical(metadata))
            if destination_mode(target) != plan["destination_mode"]:
                raise ValueError("Destination changed during import; refusing to replace it")
            progress({"phase": "publishing", "step": total, "max_steps": total})
            if plan["destination_mode"] == "new":
                os.rename(stage, target)
            else:
                # Keep existing operational journals. Publish the catalog last, under
                # both destination locks, so no server/indexer sees a partial index.
                moved = []
                try:
                    for path in stage.iterdir():
                        if path.name == "catalog.sqlite3":
                            continue
                        output = target / path.name
                        if output.exists():
                            raise ValueError("Destination changed during publication")
                        os.rename(path, output)
                        moved.append(output)
                    os.rename(stage / "catalog.sqlite3", target / "catalog.sqlite3")
                except BaseException:
                    for path in reversed(moved):
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                    raise
            return {
                "status": "completed",
                "directory": str(target),
                "workspace_id": metadata["id"],
                "import": plan,
            }
        finally:
            if stage.exists():
                shutil.rmtree(stage)
