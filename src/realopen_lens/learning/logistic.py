"""Small, explicit JSON logistic models; no training dependencies or pickle at runtime."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from realopen_lens.learning.features import NAMES, VERSION, active_indices, matrix


@dataclass(frozen=True)
class LogisticReranker:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    compatibility: dict
    sha256: str
    scope: str = "union"
    profile: str = "full"

    def scores(self, candidates: list[dict]) -> np.ndarray:
        """Return logits; monotonic sigmoid is unnecessary for ranking or calibration claims."""
        with np.errstate(over="raise", invalid="raise"):
            try:
                values = ((matrix(candidates) - self.mean) / self.scale) @ self.weights
                values += self.intercept
            except FloatingPointError as error:
                raise ValueError("Nonfinite reranker scores") from error
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite reranker scores")
        return values

    def rank(self, candidates: list[dict]) -> list[str]:
        keys = [row["image_id"] for row in candidates]
        if any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Reranker candidates need unique image identities")
        scores = self.scores(candidates)
        ordered = sorted(range(len(keys)), key=lambda i: (-scores[i], keys[i]))
        if self.scope == "hybrid50":
            hybrid = sorted(
                range(len(keys)), key=lambda i: (-candidates[i]["hybrid_score"], keys[i])
            )
            selected = set(hybrid[:50])
            ordered = [i for i in ordered if i in selected] + hybrid[50:]
        return [keys[i] for i in ordered]


def load(path: Path) -> LogisticReranker:
    """Fail explicitly for incompatible, oversized, or malformed local model files."""
    with path.open("rb") as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("Reranker model exceeds 1 MiB")
    try:
        model = json.loads(payload)
        if (
            type(model["version"]) is not int
            or model["version"] not in (1, 2)
            or model["kind"] != "logistic_regression"
            or model["feature_schema"] != VERSION
            or model["features"] != list(NAMES)
            or model["candidate_budget"] != 50
        ):
            raise ValueError("Incompatible reranker schema or candidate budget")
        arrays = [np.asarray(model[key], dtype=np.float64) for key in ("mean", "scale", "weights")]
        intercept = float(model["intercept"])
        if (
            any(value.shape != (len(NAMES),) or not np.isfinite(value).all() for value in arrays)
            or (arrays[1] <= 0).any()
            or not np.isfinite(intercept)
            or not isinstance(model["compatibility"], dict)
            or set(model["compatibility"]) != {"model_registry", "inference_sha256"}
        ):
            raise ValueError("Invalid reranker parameters or encoder compatibility")
        scope, profile = "union", "full"
        if model["version"] == 2:
            scope, profile = model["rerank_scope"], model["feature_profile"]
            if scope not in ("union", "hybrid50"):
                raise ValueError("Unknown reranking scope")
            active = set(active_indices(profile))
            for i in set(range(len(NAMES))) - active:
                if arrays[0][i] != 0 or arrays[1][i] != 1 or arrays[2][i] != 0:
                    raise ValueError("Excluded features must have neutral parameters")
        return LogisticReranker(
            arrays[0],
            arrays[1],
            arrays[2],
            intercept,
            model["compatibility"],
            hashlib.sha256(payload).hexdigest(),
            scope,
            profile,
        )
    except (KeyError, TypeError, OverflowError, UnicodeDecodeError) as error:
        raise ValueError("Malformed reranker model") from error
