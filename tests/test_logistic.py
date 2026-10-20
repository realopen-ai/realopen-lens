"""Focused regression coverage for experimental reranking."""

import numpy as np
import pytest
from realopen_lens.learning.features import NAMES, matrix


def candidate(key, visual):
    return {
        "image_id": key,
        "visual_cosine": visual,
        "semantic_cosine": None,
        "lexical_evidence": 0.0,
        "semantic_strength": 0.0,
        "hybrid_score": visual,
        "lexical_bm25": None,
        "channel_ranks": {"visual": 1, "semantic": None, "lexical": None},
        "ocr_characters": 0,
    }


def test_features_ignore_labels_identities_and_handle_missing_evidence():
    row = candidate("a", 0.7)
    row.update(label=1, grade=2, split="test", query_family="secret", category="documents")
    original = matrix([row])
    row.update(label=0, grade=0, image_id="another", split="train", query_family="other")
    np.testing.assert_array_equal(matrix([row]), original)
    assert original.shape == (1, len(NAMES)) and np.isfinite(original).all()
    assert original[0, NAMES.index("semantic_missing")] == 1
    row["lexical_bm25"] = -2
    assert matrix([row])[0, NAMES.index("bm25_log")] == pytest.approx(np.log(3))
    row["visual_cosine"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        matrix([row])
