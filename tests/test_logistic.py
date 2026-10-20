"""Focused regression coverage for experimental reranking."""

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.datasets.downloads import digest
from benchmarks.datasets.relevance import load as load_judgments
from benchmarks.datasets.training_review import apply
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


def test_frozen_review_only_adds_explicit_training_pairs():
    base = Path(__file__).resolve().parents[1] / "benchmarks/sources/retrieval-v1"
    corpus, original = load_judgments(base / "corpus.json", base / "judgments.json")
    review = json.loads((base / "training-review-v1.json").read_text(encoding="utf-8"))
    assert review["corpus_sha256"] == digest(base / "corpus.json")
    assert review["judgments_sha256"] == digest(base / "judgments.json")
    updated = apply(original, corpus, review)
    expected = {(row["family"], row["image_id"]): row for row in review["pairs"]}
    additions = 0
    for old, new in zip(original["queries"], updated["queries"]):
        if old["split"] != "train":
            assert old == new
        else:
            assert new["judgments"][: len(old["judgments"])] == old["judgments"]
            for label in new["judgments"][len(old["judgments"]) :]:
                assert label["grade"] == expected[old["family"], label["image_sha256"]]["grade"]
                assert label["grade"] is not None
                additions += 1
    assert additions == 507
    assert sum(row["grade"] is None for row in review["pairs"]) == 43
    invalid = deepcopy(review)
    invalid["pairs"][0]["family"] = next(
        q["family"] for q in original["queries"] if q["split"] == "test"
    )
    with pytest.raises(ValueError, match="held-out"):
        apply(original, corpus, invalid)
