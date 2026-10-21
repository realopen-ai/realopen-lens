"""Deterministic bounded Newton fitting for a small L2-regularized logistic baseline."""

from collections import Counter
from collections.abc import Callable
import numpy as np

from realopen_lens.learning.features import NAMES, VERSION, active_indices, matrix


def query_weights(rows: list[dict]) -> np.ndarray:
    """Equal family mass, equal translation mass, balanced classes within each query."""
    counts = Counter((row["query_id"], row["label"]) for row in rows)
    languages: dict[str, set] = {}
    for row in rows:
        languages.setdefault(row["query_family"], set()).add(row["query_id"])
    weights = []
    for row in rows:
        key, label = row["query_id"], row["label"]
        classes = sum((key, value) in counts for value in (0, 1))
        weights.append(1 / (len(languages[row["query_family"]]) * classes * counts[key, label]))
    result = np.asarray(weights, dtype=np.float64)
    return result / result.sum()


def fit(
    rows: list[dict],
    l2: float,
    compatibility: dict,
    max_steps: int = 100,
    *,
    profile: str | None = None,
    scope: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Fit train-only scaling and weighted log loss + L2/2 ||weights||², bias unpenalized."""
    if not rows or not np.isfinite(l2) or l2 <= 0 or not 1 <= max_steps <= 1000:
        raise ValueError("Training needs rows, positive finite regularization, and bounded steps")
    if any(
        row["query_split"] != "train"
        or row["image_split"] != "train"
        or row["label"] not in (0, 1)
        or not row["training_eligible"]
        for row in rows
    ):
        raise ValueError("Fit accepts only explicitly labeled same-split training pairs")
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    if len(set(labels)) != 2:
        raise ValueError("Logistic training requires both positive and negative labels")
    version = 2 if profile is not None or scope is not None else 1
    if "siglip_precision" in compatibility:
        version = 3
    profile, scope = profile or "full", scope or "union"
    if scope not in ("union", "hybrid50"):
        raise ValueError("Unknown reranking scope")
    active = active_indices(profile)
    values = matrix([row["features"] for row in rows])
    if len(active) != len(NAMES):
        values = values[:, active]
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack(((values - mean) / scale, np.ones(len(rows))))
    sample = query_weights(rows)
    coefficients = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], l2)
    penalty[-1] = 0

    def objective(beta):
        logits = design @ beta
        return float(
            sample @ (np.logaddexp(0, logits) - labels * logits) + 0.5 * (penalty * beta) @ beta
        )

    losses = [objective(coefficients)]
    for step in range(max_steps):
        if on_progress is not None:
            on_progress(
                {"phase": "training", "step": step, "max_steps": max_steps, "loss": losses[-1]}
            )
        logits = design @ coefficients
        probabilities = np.exp(-np.logaddexp(0, -logits))
        gradient = design.T @ (sample * (probabilities - labels)) + penalty * coefficients
        residual = float(np.max(np.abs(gradient)))
        if residual < 1e-7:
            break
        curvature = sample * probabilities * (1 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + np.diag(penalty + 1e-12)
        direction = np.linalg.solve(hessian, gradient)
        rate = 1.0
        for _ in range(50):
            proposal = coefficients - rate * direction
            value = objective(proposal)
            if np.isfinite(value) and value <= losses[-1] - 1e-4 * rate * gradient @ direction:
                break
            rate *= 0.5
        else:
            raise ValueError("Logistic optimizer line search did not converge")
        coefficients = proposal
        losses.append(value)
    else:
        raise ValueError("Logistic optimizer exceeded its iteration limit")
    full_mean, full_scale, full_weights = (
        np.zeros(len(NAMES)),
        np.ones(len(NAMES)),
        np.zeros(len(NAMES)),
    )
    full_mean[active], full_scale[active], full_weights[active] = mean, scale, coefficients[:-1]
    result = {
        "version": version,
        "kind": "logistic_regression",
        "status": "experimental",
        "feature_schema": VERSION,
        "features": list(NAMES),
        "candidate_budget": 50,
        "mean": full_mean.tolist(),
        "scale": full_scale.tolist(),
        "weights": full_weights.tolist(),
        "intercept": float(coefficients[-1]),
        "compatibility": compatibility,
        "training": {
            "l2": l2,
            "rows": len(rows),
            "positives": int(labels.sum()),
            "families": len({row["query_family"] for row in rows}),
            "steps": step,
            "gradient_max_abs": residual,
            "loss_history": losses,
            "weighting": "equal family/translation mass and per-query balanced classes",
        },
    }
    if version in (2, 3):
        result.update(feature_profile=profile, rerank_scope=scope)
    return result
