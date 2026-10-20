"""Versioned, label-free features shared by training and production candidate ranking."""

import math
import numpy as np

VERSION = 1
NAMES = (
    "visual_cosine",
    "semantic_cosine",
    "semantic_missing",
    "lexical_evidence",
    "semantic_strength",
    "bm25_log",
    "lexical_missing",
    "visual_reciprocal_rank",
    "semantic_reciprocal_rank",
    "lexical_reciprocal_rank",
    "hybrid_score",
    "log_ocr_characters",
    "visual_lexical_interaction",
    "visual_semantic_interaction",
)

EXCLUDED = {
    "full": (),
    "no-bm25": ("bm25_log",),
    "no-ocr-length": ("log_ocr_characters",),
    "no-missing": ("semantic_missing", "lexical_missing"),
    "core": ("bm25_log", "log_ocr_characters", "semantic_missing", "lexical_missing"),
}


def active_indices(profile: str) -> list[int]:
    if profile not in EXCLUDED:
        raise ValueError("Unknown reranker feature profile")
    return [i for i, name in enumerate(NAMES) if name not in EXCLUDED[profile]]


def matrix(candidates: list[dict]) -> np.ndarray:
    """Read an explicit allowlist, ignoring identities, labels, and arbitrary extra fields."""
    rows = []
    for row in candidates:
        visual = float(row["visual_cosine"])
        semantic = row["semantic_cosine"]
        lexical = float(row["lexical_evidence"])
        bm25 = row["lexical_bm25"]
        characters = row["ocr_characters"]
        if characters < 0 or (bm25 is not None and bm25 > 0):
            raise ValueError("Invalid OCR length or lower-is-better BM25 score")
        ranks = row["channel_ranks"]
        reciprocal = []
        for mode in ("visual", "semantic", "lexical"):
            rank = ranks[mode]
            if rank is not None and (type(rank) is not int or not 1 <= rank <= 50):
                raise ValueError("Channel ranks must be absent or within the top 50")
            reciprocal.append(1 / rank if rank is not None else 0.0)
        semantic_value = float(semantic) if semantic is not None else 0.0
        rows.append(
            [
                visual,
                semantic_value,
                float(semantic is None),
                lexical,
                float(row["semantic_strength"]),
                math.log1p(-bm25) if bm25 is not None else 0.0,
                float(bm25 is None),
                *reciprocal,
                float(row["hybrid_score"]),
                math.log1p(characters),
                visual * lexical,
                visual * semantic_value,
            ]
        )
    values = np.asarray(rows, dtype=np.float64).reshape(-1, len(NAMES))
    if not np.isfinite(values).all():
        raise ValueError("Reranker features must be finite")
    return values
