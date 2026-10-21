"""Reproducible logistic model selection and immutable local run artifacts."""

from datetime import UTC, datetime
import json
from pathlib import Path
import platform
from typing import Any
from time import perf_counter

import numpy as np

from realopen_lens.learning.dataset import canonical, fingerprint, validate_frozen
from realopen_lens.learning.evaluation import aggregate, production, replay
from realopen_lens.learning.logistic import load
from realopen_lens.learning.trainer import fit
from realopen_lens.models.manager import digest
from realopen_lens.storage.artifacts import atomic_bytes
from realopen_lens.storage.catalog import identifier


def train(
    dataset_path: Path, index_dir: Path, models: Path = Path("models"), on_progress=None
) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    validate_frozen(dataset)
    frozen = json.loads(dataset_path.with_name("freeze.json").read_text(encoding="utf-8"))
    if frozen.get("version") != 1 or any(
        digest(dataset_path if name == "dataset" else dataset_path.with_name(name + suffix))
        != frozen.get(name + "_sha256")
        for name, suffix in (("dataset", ".json"), ("review", ".json"), ("feedback", ".jsonl"))
    ):
        raise ValueError(
            "Frozen dataset, review or feedback snapshot changed; prepare a new dataset"
        )
    if dataset.get("version") != 1 or not dataset.get("sessions"):
        raise ValueError("Expected a frozen version-1 reviewed dataset")
    # The prepared artifact is an input, never a writable training output.
    run_id = identifier()
    directory = index_dir / "rerankers" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    from realopen_lens.learning.registry import runtime_identity

    identity: dict[str, Any] = {
        "runtime_identity": runtime_identity(),
        "version": 1,
        "id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": digest(dataset_path),
        "dataset_fingerprint": fingerprint(dataset),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "optimizer_sha256": digest(Path(__file__).with_name("trainer.py")),
        "status": "running",
        "selection": "validation nDCG@10, MRR@10, then stronger L2",
        "l2_grid": [0.001, 0.01, 0.1, 1.0],
        "rerank_scope": "hybrid50",
    }
    atomic_bytes(directory / "run.json", canonical(identity))
    atomic_bytes(directory / "dataset.json", canonical(dataset))
    for name in ("review.json", "feedback.jsonl", "freeze.json"):
        atomic_bytes(directory / name, dataset_path.with_name(name).read_bytes())
    started = perf_counter()
    try:
        rows = [
            {
                "query_id": session["run_id"],
                "query_family": session["family"],
                "query_split": "train",
                "image_split": "train",
                "training_eligible": True,
                "label": session["labels"][row["image_id"]],
                "features": row,
            }
            for session in dataset["sessions"]
            if session["split"] == "train"
            for row in session["candidates"]
            if row["image_id"] in session["labels"]
        ]
        validation = [s for s in dataset["sessions"] if s["split"] == "validation"]
        test = [s for s in dataset["sessions"] if s["split"] == "test"]
        if not validation or not test:
            raise ValueError("Frozen validation and test queries are required")
        trials: list[dict] = []
        for l2 in identity["l2_grid"]:
            if on_progress:
                on_progress({"phase": "selecting", "trial": len(trials) + 1, "trials": 4})
            artifact = fit(
                rows,
                l2,
                dataset["compatibility"],
                profile="core",
                scope="hybrid50",
                on_progress=on_progress,
            )
            path = directory / f"trial-{l2}.json"
            atomic_bytes(path, canonical(artifact))
            scores = aggregate(replay(load(path), validation))
            trials.append({"l2": l2, "path": path.name, "validation": scores})
        selected = max(
            trials,
            key=lambda t: (
                t["validation"]["logistic"]["ndcg_at_10"],
                t["validation"]["logistic"]["mrr_at_10"],
                t["l2"],
            ),
        )
        model = directory / "model.json"
        atomic_bytes(model, (directory / selected["path"]).read_bytes())
        # Frozen selection is persisted before opening test evaluation.
        atomic_bytes(
            directory / "selection.json", canonical({"selected": selected, "trials": trials})
        )
        report = production(
            dataset, model, index_dir, models, directory / "queries.jsonl", on_progress
        )
        atomic_bytes(directory / "evaluation.json", canonical(report))
        identity.update(
            status="completed" if report["failed"] == 0 else "completed_with_errors",
            model_sha256=digest(model),
            evaluation_sha256=digest(directory / "evaluation.json"),
            selected_l2=selected["l2"],
            evaluation=report,
            seconds=perf_counter() - started,
        )
    except BaseException as error:
        identity.update(
            status="interrupted" if not isinstance(error, Exception) else "failed",
            error=f"{type(error).__name__}: {error}",
            seconds=perf_counter() - started,
        )
        atomic_bytes(directory / "run.json", canonical(identity))
        raise
    atomic_bytes(directory / "run.json", canonical(identity))
    return identity
