"""Resumable duplicate scans and non-destructive review over published generations."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Callable

from realopen_lens.models.manager import digest
from realopen_lens.organization.fingerprints import POLICY, fingerprint, near_distance
from realopen_lens.runtime.jobs import writer_lock
from realopen_lens.storage.artifacts import atomic_bytes, load_manifest
from realopen_lens.storage.catalog import Catalog


def _save(path: Path, value: dict):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _key(manifest: dict) -> str:
    # Includes signatures: an index refresh cannot reuse an outdated scan by accident.
    payload = {"policy": POLICY, "pipeline": manifest["pipeline"], "images": manifest["images"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def snapshot(root: Path, manifest: dict) -> dict | None:
    path = root / "duplicates" / (_key(manifest) + ".json")
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("policy") != POLICY or value.get("key") != _key(manifest):
        raise ValueError("Duplicate snapshot is incompatible; scan duplicates again")
    return value


def exact_groups(root: Path, manifest: dict, paths: dict) -> dict[str, tuple[str, ...]]:
    found = snapshot(root, manifest)
    groups = {}
    if found:
        for group in found["exact"]:
            ids = tuple(key for key in group["image_ids"] if key in paths)
            for key in ids:
                groups[key] = ids
    return groups


def collapse(ranking: list[str], groups: dict[str, tuple[str, ...]]) -> list[str]:
    seen: set[str] = set()
    result = []
    for key in ranking:
        if key not in seen:
            result.append(key)
            seen.update(groups.get(key, (key,)))
    return result


def _cached(path: Path, key: str) -> dict | None:
    """Treat damaged derived cache entries as misses, never authoritative identities."""
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("policy") != POLICY or cached.get("image_id") != key:
            return None
        value = cached["fingerprint"]
        for field, size in (("pixels", 64), ("dhash", 64), ("rgb16", 1536)):
            if len(value[field]) != size or len(bytes.fromhex(value[field])) != size // 2:
                return None
        if (
            type(value["width"]) is not int
            or type(value["height"]) is not int
            or not 0 < value["width"] * value["height"] <= 40_000_000
            or min(value["width"], value["height"]) <= 0
            or not math.isfinite(value["contrast"])
        ):
            return None
        return value
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return None


def scan(root: Path, collection: str = "personal", on_progress: Callable | None = None) -> dict:
    """Cache each success, publish atomically, and leave old complete snapshots intact."""
    from realopen_lens.retrieval.search import source_paths

    started = perf_counter()
    with writer_lock(root):
        catalog = Catalog(root, readonly=True)
        try:
            found, active = catalog.snapshot(collection)
            manifest = load_manifest(catalog, active)
            paths = source_paths(Path(found["root"]), manifest)
            records, errors, unsupported = {}, [], []
            reused = 0

            def progress(phase: str, **extra):
                if on_progress:
                    on_progress({"phase": phase, **extra})

            for position, row in enumerate(manifest["images"], 1):
                key = row["image_id"]
                progress("fingerprinting", step=position - 1, max_steps=len(manifest["images"]))
                if key not in paths:
                    errors.append({"image_id": key, "error": "Source unavailable or changed"})
                    continue
                path = Path(paths[key][0])
                cache = root / "duplicates" / "cache" / (key + ".json")
                try:
                    if digest(path) != key:
                        raise ValueError("Source bytes differ from indexed identity")
                    cached = _cached(cache, key)
                    if cached is not None:
                        data = cached
                        reused += 1
                    else:
                        data = fingerprint(path)
                    if digest(path) != key:
                        raise ValueError("Source changed during fingerprinting")
                    _save(cache, {"policy": POLICY, "image_id": key, "fingerprint": data})
                    records[key] = data
                except Exception as error:
                    item = {"image_id": key, "error": str(error)[:1000]}
                    if isinstance(error, ValueError) and "Multi-frame" in str(error):
                        unsupported.append(item)
                    else:
                        errors.append(item)
            progress("grouping", step=len(records), max_steps=len(manifest["images"]))
            buckets: dict[str, list[str]] = {}
            for key, value in records.items():
                buckets.setdefault(value["pixels"], []).append(key)
            exact = [
                {"id": key, "image_ids": sorted(ids)}
                for key, ids in sorted(buckets.items())
                if len(ids) > 1
            ]
            # A BK tree bounds the candidate work without allocating an NxN matrix.
            tree: dict = {}
            pairs = []
            comparisons = 0
            texts = {
                row["image_id"]: " ".join(row["text"].casefold().split()) for row in manifest["ocr"]
            }
            ids = sorted(min(values) for values in buckets.values())
            for position, key in enumerate(ids):
                progress("comparing", step=position, max_steps=len(ids))
                hash_value = int(records[key]["dhash"], 16)
                stack = [tree] if tree else []
                candidates = []
                while stack:
                    node = stack.pop()
                    distance = (hash_value ^ node["hash"]).bit_count()
                    if distance <= 8:
                        candidates.extend(node["ids"])
                    stack.extend(
                        child
                        for edge, child in node["children"].items()
                        if distance - 8 <= edge <= distance + 8
                    )
                for other in candidates:
                    comparisons += 1
                    if comparisons > 5_000_000:
                        raise ValueError("Duplicate comparison budget exceeded (5 million pairs)")
                    distance = near_distance(records[key], records[other])
                    if distance is not None:
                        pair_ids = sorted(
                            buckets[records[key]["pixels"]] + buckets[records[other]["pixels"]]
                        )
                        pair_id = hashlib.sha256(
                            (POLICY + ":" + ":".join(pair_ids)).encode()
                        ).hexdigest()
                        pairs.append(
                            {
                                "id": pair_id,
                                "image_ids": pair_ids,
                                "text_differs": texts.get(key, "") != texts.get(other, ""),
                                **distance,
                            }
                        )
                        if len(pairs) > 10000:
                            raise ValueError(
                                "Near-duplicate suggestion budget exceeded (10000 pairs)"
                            )
                node = tree
                while node:
                    distance = (hash_value ^ node["hash"]).bit_count()
                    if distance == 0:
                        node["ids"].append(key)
                        break
                    node = node["children"].setdefault(distance, {})
                if not node:
                    node.update(hash=hash_value, ids=[key], children={})
            # Refuse to publish results if source generation or signatures drifted.
            _, current = catalog.snapshot(collection)
            if (
                current["id"] != active["id"]
                or source_paths(Path(found["root"]), manifest) != paths
            ):
                raise ValueError("Source changed during duplicate scan; refresh and retry")
            result = {
                "policy": POLICY,
                "key": _key(manifest),
                "generation": active["id"],
                "collection": collection,
                "images": len(manifest["images"]),
                "fingerprinted": len(records),
                "reused": reused,
                "errors": errors,
                "unsupported": unsupported,
                "pixels": {key: value["pixels"] for key, value in records.items()},
                "exact": exact,
                "near": pairs,
                "seconds": perf_counter() - started,
                "status": "completed_with_errors" if errors else "completed",
            }
            progress("publishing", step=len(ids), max_steps=len(ids))
            _save(root / "duplicates" / (result["key"] + ".json"), result)
            return result
        finally:
            catalog.close()


def review(root: Path, collection: str, pair_id: str, state: str) -> dict:
    if state not in {"confirmed", "dismissed", "unreviewed"}:
        raise ValueError("Review must be confirmed, dismissed or unreviewed")
    with writer_lock(root):
        catalog = Catalog(root, readonly=True)
        try:
            found, active = catalog.snapshot(collection)
            data = snapshot(root, load_manifest(catalog, active))
            if not data or not any(pair["id"] == pair_id for pair in data["near"]):
                raise ValueError("Suggestion not found in the current duplicate scan")
            path = root / "duplicates" / "reviews" / (found["id"] + ".json")
            values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            values[pair_id] = state
            _save(path, values)
            return {"id": pair_id, "state": state}
        finally:
            catalog.close()


def browse(root: Path, collection: str = "personal", offset: int = 0, limit: int = 20) -> dict:
    from realopen_lens.retrieval.search import source_paths, search_results

    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("Invalid duplicate page range")
    catalog = Catalog(root, readonly=True)
    try:
        found, active = catalog.snapshot(collection)
        manifest = load_manifest(catalog, active)
        data = snapshot(root, manifest)
        if data is None:
            return {"scanned": False, "groups": [], "total": 0, "generation": active["id"]}
        path = root / "duplicates" / "reviews" / (found["id"] + ".json")
        reviews = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        paths = source_paths(Path(found["root"]), manifest)
        groups = [{**g, "kind": "exact", "state": "exact"} for g in data["exact"]]
        groups += [
            {**g, "kind": "near", "state": reviews.get(g["id"], "unreviewed")} for g in data["near"]
        ]
        for group in groups[offset : offset + limit]:
            ids = [key for key in group["image_ids"] if key in paths]
            group["results"] = [
                asdict(r)
                for r in search_results(catalog, manifest, paths, ids, {}, deduplicate=False)
            ]
            group["unavailable"] = len(group["image_ids"]) - len(ids)
        return {
            "scanned": True,
            "generation": active["id"],
            "total": len(groups),
            "groups": groups[offset : offset + limit],
            "summary": {
                key: data[key]
                for key in ("images", "fingerprinted", "reused", "errors", "unsupported", "seconds")
            },
        }
    finally:
        catalog.close()
