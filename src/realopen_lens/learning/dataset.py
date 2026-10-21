"""Freeze explicitly reviewed feedback, preserving group and encoder boundaries."""

import hashlib
import json
from pathlib import Path
import unicodedata

from realopen_lens.learning.features import matrix
from realopen_lens.learning.feedback import export_feedback
from realopen_lens.models.manager import digest
from realopen_lens.storage.artifacts import atomic_bytes
from realopen_lens.storage.feedback import FeedbackStore

SPLITS = {"train", "validation", "test"}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _prepare(index_dir: Path, review: Path, output: Path, on_progress=None) -> dict:
    """Review binds exact export bytes and explicitly assigns families/source groups.

    No automatic independence claims or inferred negative labels. Existing output
    directories are refused so training input snapshots cannot be silently replaced.
    """
    if output.exists():
        raise ValueError("Use a new dataset directory")
    plan = json.loads(review.read_text(encoding="utf-8"))
    if plan.get("version") != 1 or not str(plan.get("reviewer", "")).strip():
        raise ValueError("Review requires version 1 and a reviewer")
    source = review.parent / plan["feedback_file"]
    if digest(source) != plan["feedback_sha256"]:
        raise ValueError("Review identifies a different feedback export")
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    selected = plan["runs"]
    if not selected or len(rows) > 100_000:
        raise ValueError("Select reviewed runs; at most 100,000 feedback pairs")
    rows = [row for row in rows if row["run_id"] in selected]
    if {row["run_id"] for row in rows} != set(selected):
        raise ValueError("Every selected run must have explicit labels")
    families: dict[str, str] = {}
    groups: dict[str, str] = {}
    queries: dict[str, str] = {}
    pairs: set[tuple[str, str]] = set()
    compatibility = None
    sources = None
    sessions = []
    with FeedbackStore(index_dir, readonly=True) as store:
        store.db.execute("BEGIN")
        for position, (run_id, assignment) in enumerate(sorted(selected.items())):
            if on_progress:
                on_progress({"phase": "preparing", "completed": position, "total": len(selected)})
            split, family = assignment["split"], assignment["family"]
            if split not in SPLITS or not isinstance(family, str) or not family.strip():
                raise ValueError("Each run needs an explicit split and query family")
            if family in families and families[family] != split:
                raise ValueError("Query family leaks across splits")
            families[family] = split
            run = store.run(run_id)
            if run["query"]["kind"] != "text" or run["ranking"]["mode"] != "hybrid":
                raise ValueError("Training supports text hybrid searches only")
            if run.get("scope") is not None:
                raise ValueError(
                    "Scoped feedback requires a scope-aware evaluation review; "
                    "use unscoped runs for training"
                )
            ranking = run["ranking"]
            if (
                ranking["siglip_precision"] != run["encoder_spec"].get("siglip_precision", "int8")
                or ranking.get("reranker_sha256")
                or ranking.get("candidate_depths") != dict(visual=50, semantic=50, lexical=50)
            ):
                raise ValueError(
                    "Use unreranked precision-compatible hybrid searches with 50/channel"
                )
            query = unicodedata.normalize("NFKC", run["query"]["text"]).casefold().strip()
            if query in queries:
                raise ValueError("Select only one run per normalized query")
            queries[query] = run_id
            spec = run["encoder_spec"]
            current = {key: spec[key] for key in ("model_registry", "inference_sha256")}
            current["siglip_precision"] = spec.get("siglip_precision", "int8")
            if compatibility is not None and current != compatibility:
                raise ValueError("Mixed encoder compatibility in feedback")
            compatibility = current
            if sources is not None and sources != run["ranking_sources"]:
                raise ValueError("Mixed ranking source versions in feedback")
            sources = run["ranking_sources"]
            candidates = list(run["candidates"].values())
            matrix(candidates)
            labels = {}
            active = {r["image_id"]: r for r in store.judgments(run_id)}
            for row in (r for r in rows if r["run_id"] == run_id):
                key = row["image_id"]
                if (run_id, key) in pairs:
                    raise ValueError("Duplicate feedback pair")
                pairs.add((run_id, key))
                if (
                    row["version"] != 1
                    or type(row["label"]) is not int
                    or row["label"] not in (0, 1)
                    or row.get("candidate") != run["candidates"].get(key)
                    or key not in run["candidates"]
                    or row["query"] != run["query"]
                    or row["encoder_spec"] != spec
                    or row["ranking_sources"] != sources
                    or row["ranking"] != ranking
                    or row["generation"] != run["generation"]
                    or key not in active
                    or active[key]["revision"] != row["revision"]
                    or active[key]["label"] != ("relevant" if row["label"] else "irrelevant")
                ):
                    raise ValueError("Feedback changed or export differs from saved provenance")
                group = plan["image_groups"][key]
                if not isinstance(group, str) or not group.strip():
                    raise ValueError("Each labeled image needs a reviewed source/duplicate group")
                if group in groups and groups[group] != split:
                    raise ValueError("Source/duplicate group leaks across splits")
                groups[group] = split
                labels[key] = row["label"]
            if not any(labels.values()):
                raise ValueError("Every query needs a reviewed positive")
            sessions.append(
                {
                    "run_id": run_id,
                    "query": run["query"],
                    "split": split,
                    "family": family,
                    "image_groups": {key: plan["image_groups"][key] for key in labels},
                    "candidates": candidates,
                    "labels": labels,
                    "generation": run["generation"],
                    "collection": run["collection"],
                }
            )
    if {s["split"] for s in sessions} != SPLITS:
        raise ValueError("Review must include train, validation and test queries")
    dataset = {
        "version": 1,
        "compatibility": compatibility,
        "ranking_sources": sources,
        "review_sha256": digest(review),
        "feedback_sha256": digest(source),
        "reviewer": plan["reviewer"],
        "independent_holdout": plan.get("independent_holdout") is True,
        "evaluation_scope": "saved displayed-candidate replay; known explicit judgments only",
        "sessions": sessions,
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_bytes(output / "dataset.json", canonical(dataset))
    atomic_bytes(output / "feedback.jsonl", source.read_bytes())
    atomic_bytes(output / "review.json", review.read_bytes())
    atomic_bytes(
        output / "freeze.json",
        canonical(
            {
                "version": 1,
                "dataset_sha256": digest(output / "dataset.json"),
                "feedback_sha256": digest(output / "feedback.jsonl"),
                "review_sha256": digest(output / "review.json"),
            }
        ),
    )
    return {
        "dataset": str(output),
        "sha256": fingerprint(dataset),
        "queries": len(sessions),
        "pairs": len(rows),
        "independent_holdout": dataset["independent_holdout"],
    }


def export_snapshot(index_dir: Path, output: Path) -> int:
    return export_feedback(output, index_dir=index_dir)


def validate_frozen(dataset: dict) -> None:
    """Recheck isolation at training time, including manually edited artifacts."""
    try:
        sessions = dataset["sessions"]
        if dataset["version"] != 1 or {s["split"] for s in sessions} != SPLITS:
            raise ValueError("Frozen dataset needs train, validation and test queries")
        families: dict[str, str] = {}
        groups: dict[str, str] = {}
        queries = set()
        images: dict[str, str] = {}
        for session in sessions:
            split = session["split"]
            family = session["family"]
            if not family or (family in families and families[family] != split):
                raise ValueError("Frozen query family leaks across splits")
            families[family] = split
            query = unicodedata.normalize("NFKC", session["query"]["text"]).casefold().strip()
            if query in queries:
                raise ValueError("Frozen dataset repeats a normalized query")
            queries.add(query)
            ids = [c["image_id"] for c in session["candidates"]]
            labels = session["labels"]
            if (
                len(ids) != len(set(ids))
                or not set(labels) <= set(ids)
                or not any(labels.values())
                or any(type(v) is not int or v not in (0, 1) for v in labels.values())
            ):
                raise ValueError("Frozen candidate labels are invalid")
            matrix(session["candidates"])
            for key in labels:
                if key in images and images[key] != split:
                    raise ValueError("Frozen image identity leaks across splits")
                images[key] = split
                group = session["image_groups"][key]
                if not group or (group in groups and groups[group] != split):
                    raise ValueError("Frozen source/duplicate group leaks across splits")
                groups[group] = split
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed frozen dataset") from error


def prepare(index_dir: Path, review: Path, output: Path, on_progress=None) -> dict:
    try:
        return _prepare(index_dir, review, output, on_progress)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed feedback review or export") from error
