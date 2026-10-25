"""Read-only feedback cohort audit and reviewed duplicate split constraints."""

from collections import Counter
import json
from pathlib import Path

from realopen_lens.learning.dataset import fingerprint
from realopen_lens.organization.duplicates import snapshot
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog
from realopen_lens.storage.feedback import FeedbackStore


def duplicate_constraints(index_dir: Path, collection: str) -> dict:
    """Confirmed similarity constrains isolation; it never transfers relevance labels."""
    catalog = Catalog(index_dir, readonly=True)
    try:
        found, active = catalog.snapshot(collection)
        manifest = load_manifest(catalog, active)
    finally:
        catalog.close()
    data = snapshot(index_dir, manifest)
    if data is None:
        return {"version": 1, "collection_id": found["id"], "components": {}, "evidence": None}
    path = index_dir / "duplicates" / "reviews" / (found["id"] + ".json")
    raw = path.read_bytes() if path.exists() else b"{}"
    reviews = json.loads(raw)
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root
            key = next_key
        return root

    used = []
    for kind, groups in (("exact", data["exact"]), ("confirmed", data["near"])):
        for group in groups:
            if kind == "confirmed" and reviews.get(group["id"]) != "confirmed":
                continue
            ids = group["image_ids"]
            for key in ids[1:]:
                a, b = find(ids[0]), find(key)
                parent[max(a, b)] = min(a, b)
            used.append({"kind": kind, "id": group["id"], "image_ids": ids})
    import hashlib

    return {
        "version": 1,
        "collection_id": found["id"],
        "generation": active["id"],
        "components": {key: find(key) for key in sorted(parent)},
        "evidence": {
            "snapshot_key": data["key"],
            "policy": data["policy"],
            "review_sha256": hashlib.sha256(raw).hexdigest(),
            "groups": used,
        },
    }


def audit(index_dir: Path, collection: str = "personal") -> dict:
    """Describe eligibility and exact provenance cohorts without mutating feedback."""
    catalog = Catalog(index_dir, readonly=True)
    try:
        found, active = catalog.snapshot(collection)
        manifest = load_manifest(catalog, active)
    finally:
        catalog.close()
    cohorts: dict[str, dict] = {}
    runs = []
    with FeedbackStore(index_dir, readonly=True) as store:
        store.db.execute("BEGIN")
        for row in store.db.execute("SELECT id,payload FROM searches ORDER BY rowid"):
            saved = json.loads(row["payload"])
            if saved["collection_id"] != found["id"]:
                continue
            judgments = store.judgments(row["id"])
            if not judgments:
                continue
            counts = dict(Counter(j["label"] for j in judgments))
            reasons = []
            from realopen_lens.learning.feedback import superseded_judgments

            if superseded_judgments(store, index_dir, {**saved, "run_id": row["id"]}):
                reasons.append("superseded judgments; review current labels")
            ranking = saved["ranking"]
            if ranking.get("feedback_memory", {}).get("applied"):
                reasons.append("query memory applied; collect baseline feedback")
            if saved["query"]["kind"] != "text" or ranking["mode"] != "hybrid":
                reasons.append("requires text hybrid search")
            if (ranking.get("query_interpretation") or {}).get("constraints"):
                reasons.append("metadata-filtered feedback requires filter-aware evaluation")
            if saved.get("scope"):
                reasons.append("requires full-library scope")
            if ranking.get("reranker_sha256"):
                reasons.append("already reranked")
            if ranking.get("candidate_depths") != {"visual": 50, "semantic": 50, "lexical": 50}:
                reasons.append("requires 50 candidates per channel")
            if not counts.get("relevant"):
                reasons.append("no reviewed positive")
            if saved["generation"] != active["id"]:
                reasons.append("source generation differs from current index")
            current_spec = manifest["config"]["spec"]
            compatibility = {
                key: saved["encoder_spec"][key] for key in ("model_registry", "inference_sha256")
            }
            compatibility["siglip_precision"] = saved["encoder_spec"].get(
                "siglip_precision", "int8"
            )
            if any(
                current_spec.get(key, "int8" if key == "siglip_precision" else None) != value
                for key, value in compatibility.items()
            ):
                reasons.append("encoder compatibility differs from current index")
            identity = {
                "compatibility": compatibility,
                "ranking_sources": saved["ranking_sources"],
                "duplicate_policy": {
                    key: ranking.get(key) for key in ("duplicate_policy", "duplicate_snapshot")
                },
            }
            key = fingerprint(identity)
            cohort = cohorts.setdefault(
                key,
                {
                    "id": key,
                    "identity": identity,
                    "run_ids": [],
                    "eligible_run_ids": [],
                    "labels": {"relevant": 0, "irrelevant": 0},
                },
            )
            cohort["run_ids"].append(row["id"])
            if not reasons:
                cohort["eligible_run_ids"].append(row["id"])
            for label in cohort["labels"]:
                cohort["labels"][label] += counts.get(label, 0)
            runs.append(
                {
                    "run_id": row["id"],
                    "cohort": key,
                    "query": saved["query"],
                    "labels": counts,
                    "eligible": not reasons,
                    "reasons": reasons,
                    "positive_only": not counts.get("irrelevant"),
                    "payload_sha256": fingerprint(saved),
                    "judgments_sha256": fingerprint(judgments),
                }
            )
    constraints = duplicate_constraints(index_dir, collection)
    return {
        "version": 1,
        "collection": collection,
        "generation": active["id"],
        "manifest_sha256": active["checksum"],
        "runs": runs,
        "cohorts": sorted(cohorts.values(), key=lambda c: (-len(c["eligible_run_ids"]), c["id"])),
        "duplicate_constraints": constraints,
        "notes": [
            "Unjudged images are not negatives",
            "Audit does not assign splits or attest holdout independence",
        ],
    }
