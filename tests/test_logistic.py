"""Numerical fitting, label boundaries, portable artifacts, and inference contracts."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load as load_judgments
from benchmarks.datasets.training_review import apply
from benchmarks.experiments.logistic_fit import fit
from benchmarks.experiments.logistic import run, validate_records
from realopen_lens.learning.features import NAMES, matrix
from realopen_lens.learning.logistic import load


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


def training_rows():
    return [
        {
            "query_id": "q",
            "query_family": "family",
            "query_split": "train",
            "image_split": "train",
            "training_eligible": True,
            "label": label,
            "features": candidate(str(i), value),
        }
        for i, (value, label) in enumerate([(-0.5, 0), (-0.2, 0), (0.2, 1), (0.5, 1)])
    ]


def test_logistic_fit_converges_deterministically_and_roundtrips(tmp_path):
    rows = training_rows()
    model = fit(rows, 0.1, {"model_registry": {}, "inference_sha256": "fixture"})
    assert model == fit(rows, 0.1, model["compatibility"])
    assert model["training"]["gradient_max_abs"] < 1e-7
    loss = model["training"]["loss_history"]
    assert all(b <= a for a, b in zip(loss, loss[1:])) and loss[-1] < loss[0]
    path = tmp_path / "model.json"
    save_json(path, model)
    restored = load(path)
    inputs = [row["features"] for row in rows]
    assert restored.rank(inputs) == ["3", "2", "1", "0"]
    logits = restored.scores(inputs)
    assert np.all(logits[:2] < 0) and np.all(logits[2:] > 0)
    assert restored.rank([]) == []
    assert restored.sha256 == digest(path)


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


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_split", "test"),
        ("image_split", "validation"),
        ("label", None),
        ("training_eligible", False),
    ],
)
def test_fit_rejects_held_out_and_unknown_labels(field, value):
    rows = training_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match="same-split"):
        fit(rows, 0.1, {})


def test_fit_scaling_uses_only_supplied_training_rows_and_handles_constant_columns():
    rows = training_rows()
    artifact = fit(rows, 0.1, {})
    np.testing.assert_allclose(artifact["mean"], matrix([r["features"] for r in rows]).mean(axis=0))
    assert np.isfinite(artifact["scale"]).all() and min(artifact["scale"]) > 0
    with pytest.raises(ValueError, match="both positive and negative"):
        fit([row for row in rows if row["label"]], 0.1, {})
    with pytest.raises(ValueError, match="iteration limit"):
        fit(rows, 0.001, {}, max_steps=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", [0] * len(NAMES)),
        ("weights", [float("nan")] * len(NAMES)),
        ("features", list(reversed(NAMES))),
        ("candidate_budget", 100),
        ("version", 99),
    ],
)
def test_model_rejects_invalid_parameters_and_schemas(tmp_path, field, value):
    artifact = fit(training_rows(), 0.1, {"model_registry": {}, "inference_sha256": "fixture"})
    artifact[field] = value
    path = tmp_path / "bad.json"
    save_json(path, artifact)
    with pytest.raises(ValueError):
        load(path)


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


def test_training_runner_selects_before_test_and_preserves_history(tmp_path):
    images, queries, records = [], [], []
    for split in ("train", "validation", "test"):
        positive, negative = split + "-p", split + "-n"
        for key in (positive, negative):
            images.append(
                {
                    "id": key,
                    "sha256": key,
                    "path": key,
                    "pixel_sha256": key,
                    "source_document": key,
                    "duplicate_group": key,
                    "split": split,
                    "category": "photos",
                }
            )
        query = {
            "id": split,
            "text": "صورة " + split,
            "family": split,
            "language": "ar",
            "split": split,
            "review": {
                "status": "reviewed",
                "reviewer": "fixture",
                "method": "fixture",
                "date": "2026-10-20",
            },
            "judgments": [
                {"image_sha256": positive, "grade": 2, "reason": "match"},
                {"image_sha256": negative, "grade": 0, "reason": "different"},
            ],
        }
        queries.append(query)
        records.append(
            {
                "query_id": split,
                "query": query,
                "status": "ok",
                "trace": {
                    "generation": "g",
                    "searchable_images": 6,
                    "candidates": [candidate(positive, 0.6), candidate(negative, 0.2)],
                    "rankings": {"hybrid": [positive, negative]},
                },
            }
        )
    corpus = tmp_path / "corpus.json"
    judgments = tmp_path / "judgments.json"
    review = tmp_path / "review.json"
    save_json(corpus, {"images": images})
    bundle = {"version": 1, "corpus_sha256": digest(corpus), "queries": queries}
    save_json(judgments, bundle)
    save_json(
        review,
        {
            "version": 1,
            "scope": "train-only; labels shared across translations of the same family",
            "corpus_sha256": digest(corpus),
            "judgments_sha256": digest(judgments),
            "review": queries[0]["review"],
            "pairs": [],
        },
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    config = {
        "corpus_sha256": digest(corpus),
        "judgments_sha256": digest(judgments),
        "candidate_feature_schema": 1,
        "search_limit": 50,
        "snapshot": {
            "generation": "g",
            "pipeline_spec": {"model_registry": {}, "inference_sha256": "fixture"},
        },
    }
    save_json(baseline / "config.json", config)

    def write_records():
        (baseline / "queries.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )

    write_records()
    args = SimpleNamespace(
        corpus=corpus,
        judgments=judgments,
        review=review,
        baseline_dir=baseline,
        output=tmp_path / "first",
    )
    summary = run(args)
    first = json.loads((args.output / "model.json").read_text(encoding="utf-8"))
    assert summary["queries"] == 3 and summary["training_pairs"] == 2
    assert summary["groups"]["split:test"]["logistic"]["ndcg_at_10"] == 1
    with pytest.raises(ValueError, match="new output directory"):
        run(args)
    records[-1]["trace"]["candidates"][0]["visual_cosine"] = -0.9
    records[-1]["trace"]["candidates"][1]["visual_cosine"] = 0.99
    write_records()
    args.output = tmp_path / "second"
    run(args)
    second = json.loads((args.output / "model.json").read_text(encoding="utf-8"))
    for key in ("weights", "intercept", "mean", "scale", "training"):
        assert first[key] == second[key]
    records[-1]["status"] = "error"
    with pytest.raises(ValueError, match="errors"):
        validate_records(records, bundle, {"images": images}, config)
