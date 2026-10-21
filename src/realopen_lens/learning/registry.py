"""Collection-scoped, explicit model activation and auditable rollback."""

from datetime import UTC, datetime
import json
from pathlib import Path
import re

from realopen_lens.learning.dataset import canonical
from realopen_lens.learning.logistic import load
from realopen_lens.models.manager import digest
from realopen_lens.runtime.jobs import writer_lock
from realopen_lens.storage.artifacts import atomic_bytes, load_manifest
from realopen_lens.storage.catalog import Catalog


def runtime_identity() -> dict:
    package = Path(__file__).resolve().parents[1]
    files = (
        "retrieval/search.py",
        "retrieval/ranking.py",
        "retrieval/candidates.py",
        "retrieval/identifiers.py",
        "learning/features.py",
        "learning/logistic.py",
        "learning/trainer.py",
        "learning/evaluation.py",
        "learning/workflow.py",
    )
    return {name: digest(package / name) for name in files}


def _directory(root: Path, version: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", version):
        raise ValueError("Invalid model version")
    return root / "rerankers" / version


def versions(root: Path) -> list[dict]:
    directory = root / "rerankers"
    if not directory.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*/run.json"))
    ]


def state(root: Path) -> dict:
    path = root / "rerankers" / "state.json"
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"version": 1, "active": {}, "history": []}
    )


def compatible(root: Path, collection: str, version: str) -> Path:
    directory = _directory(root, version)
    info = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    path = directory / "model.json"
    if info["status"] != "completed" or digest(path) != info["model_sha256"]:
        raise ValueError("Model must have a completed evaluation and intact artifact")
    report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    if digest(directory / "evaluation.json") != info["evaluation_sha256"]:
        raise ValueError("Model evaluation artifact changed")
    if report != info["evaluation"]:
        raise ValueError("Model evaluation summary differs from its artifact")
    if report["failed"] or report["evaluated"] == 0:
        raise ValueError("Model evaluation has failures or no evaluated queries")
    if info["runtime_identity"] != runtime_identity():
        raise ValueError(
            "Ranking/training implementation changed since evaluation; evaluate a new version"
        )
    model = load(path)
    with_catalog = Catalog(root, readonly=True)
    try:
        _, generation = with_catalog.snapshot(collection)
        spec = load_manifest(with_catalog, generation)["config"]["spec"]
    finally:
        with_catalog.close()
    if spec.get("siglip_precision", "int8") != model.compatibility.get(
        "siglip_precision", "int8"
    ) or any(spec.get(k) != v for k, v in model.compatibility.items()):
        raise ValueError("Model encoder compatibility differs from this collection")
    if model.scope != "hybrid50":
        raise ValueError("Managed models must preserve the hybrid top-50 candidate set")
    return path


def change(
    root: Path,
    collection: str,
    version: str | None,
    *,
    acknowledge_experimental: bool = False,
    rollback: bool = False,
) -> dict:
    """All changes require an explicit request. No report triggers automatic promotion."""
    with writer_lock(root / "rerankers" / "activation-lock"):
        current = state(root)
        previous = current["active"].get(collection)
        if version is not None:
            compatible(root, collection, version)
            info = json.loads((_directory(root, version) / "run.json").read_text(encoding="utf-8"))
            report = info["evaluation"]
            if not acknowledge_experimental and (
                not report["independent_holdout"] or report["regressions"]
            ):
                raise ValueError(
                    "Diagnostic holdout or regressions: "
                    "explicit experimental acknowledgment required"
                )
            if rollback and not any(
                h["collection"] == collection and h["to"] == version for h in current["history"]
            ):
                raise ValueError("Rollback target was never activated for this collection")
        if previous == version:
            return current
        if version is None:
            current["active"].pop(collection, None)
        else:
            current["active"][collection] = version
        current["history"].append(
            {
                "collection": collection,
                "from": previous,
                "to": version,
                "operation": "rollback" if rollback else "activate" if version else "disable",
                "at": datetime.now(UTC).isoformat(),
                "experimental_acknowledged": acknowledge_experimental,
            }
        )
        atomic_bytes(root / "rerankers" / "state.json", canonical(current))
        return current


def active(root: Path, collection: str) -> Path | None:
    version = state(root)["active"].get(collection)
    return compatible(root, collection, version) if version else None
