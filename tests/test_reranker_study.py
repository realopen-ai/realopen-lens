"""Recall-preserving scope, disabled features, fresh-fold guards, and source isolation."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess

from PIL import Image
import pytest

from benchmarks.datasets.downloads import digest, download, DownloadRateLimitError
from benchmarks.datasets.expansion import build, copy_verified, near_duplicate, preview, safe_path
from benchmarks.datasets.expansion_judgments import assemble
from benchmarks.datasets.relevance import load as load_judgments
from benchmarks.datasets.training_review import apply
from benchmarks.datasets.photos import save_json
from benchmarks.experiments.logistic_fit import fit
from benchmarks.experiments.reranker_study import best, guard
from realopen_lens.indexing.service import signature
from realopen_lens.learning.features import NAMES, EXCLUDED
from realopen_lens.learning.logistic import load


def candidate(index):
    return {
        "image_id": f"{index:03}",
        "visual_cosine": 1 - index / 100,
        "semantic_cosine": None,
        "lexical_evidence": 0.0,
        "semantic_strength": 0.0,
        "hybrid_score": 1 - index / 100,
        "lexical_bm25": None,
        "channel_ranks": {
            "visual": index + 1 if index < 50 else None,
            "lexical": None,
            "semantic": None,
        },
        "ocr_characters": 0,
    }


def test_frozen_v2_sources_review_and_legacy_roles_are_auditable():
    root = Path(__file__).resolve().parents[1] / "benchmarks/sources"
    base, current = root / "retrieval-v1", root / "retrieval-v2"
    corpus, bundle = load_judgments(current / "corpus.json", current / "judgments.json")
    sources = json.loads((current / "sources.json").read_text(encoding="utf-8"))
    review = json.loads((current / "review.json").read_text(encoding="utf-8"))
    limits = json.loads((current / "acquisition-limits.json").read_text(encoding="utf-8"))
    assert corpus["sources_sha256"] == digest(current / "sources.json")
    assert (
        corpus["protocol_sha256"] == bundle["protocol_sha256"] == digest(current / "protocol.json")
    )
    assert bundle["review_sha256"] == digest(current / "review.json")
    assert limits["plan_sha256"] == digest(current / "acquisition-plan.json")
    assert limits["acquired_sources"] == len(sources["images"]) == 52
    assert limits["planned_sources"] == len(sources["images"]) + len(limits["deferred"])
    original, old = load_judgments(base / "corpus.json", base / "judgments.json")
    old = apply(
        old, original, json.loads((base / "training-review-v1.json").read_text(encoding="utf-8"))
    )
    assert assemble(corpus, old, review)["queries"] == bundle["queries"]
    assert len(corpus["images"]) == 2052 and len(bundle["queries"]) == 357
    images = {r["sha256"]: r for r in corpus["images"]}
    legacy_groups = {
        images[label["image_sha256"]]["duplicate_group"]
        for q in old["queries"]
        for label in q["judgments"]
        if label["grade"] > 0
    }
    for query in bundle["queries"]:
        if query["source_exposure"] == "fresh-v2":
            assert all(
                images[label["image_sha256"]]["duplicate_group"] not in legacy_groups
                for label in query["judgments"]
                if label["grade"] > 0
            )


def test_source_acquisition_preserves_server_retry_after(tmp_path, monkeypatch):
    headers = tmp_path / "response.headers"

    def limited(command, **kwargs):
        headers.write_bytes(b"HTTP/2 429\r\nRetry-After: 600\r\n\r\n")
        raise subprocess.CalledProcessError(22, command, stderr=b"HTTP 429")

    monkeypatch.setattr(subprocess, "run", limited)
    with pytest.raises(DownloadRateLimitError) as caught:
        download("https://example.invalid/source.png", tmp_path / "image.png", headers=headers)
    assert caught.value.retry_after == 600
    assert not (tmp_path / "image.png").exists()


def test_expansion_refuses_early_resume_after_rate_limit(tmp_path, monkeypatch):
    base = tmp_path / "base.json"
    sources = tmp_path / "sources.json"
    save_json(base, {"images": []})
    save_json(
        sources,
        {
            "base_corpus_sha256": digest(base),
            "images": [
                {
                    "group": "app",
                    "split": "train",
                    "category": "screenshots",
                    "filename": "app.png",
                    "title": "App",
                    "source": {"url": "test"},
                }
            ],
        },
    )
    calls = []

    def limited(*args, **kwargs):
        calls.append(args)
        raise DownloadRateLimitError("test", 600)

    monkeypatch.setattr("benchmarks.datasets.expansion.download", limited)
    monkeypatch.setattr("benchmarks.datasets.expansion.time.sleep", lambda _: None)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="incomplete"):
        build(tmp_path, base, sources, output)
    assert (output / "download-pause.json").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(ValueError, match="requested a pause"):
        build(tmp_path, base, sources, output)
    assert len(calls) == 1


def test_dataset_paths_reject_escape_and_symlinks(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    assert safe_path(root, "photos/image.jpg") == root / "photos/image.jpg"
    for path in ("../outside.jpg", tmp_path / "outside.jpg", "."):
        with pytest.raises(ValueError, match="inside"):
            safe_path(root, path)
    if sys.platform != "win32":  # Unprivileged Windows checkouts cannot create symlinks.
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="inside"):
            safe_path(root, "link/outside.jpg")


def test_expansion_review_keeps_diagnostics_and_ambiguous_variants_out(tmp_path):
    rows = [
        {
            "path": str(i),
            "sha256": str(i),
            "split": split,
            "category": "screenshots",
            "duplicate_group": group,
        }
        for i, split, group in (
            (0, "train", "app"),
            (1, "train", "app"),
            (2, "train", "other"),
            (3, "test", "held-out"),
        )
    ]
    legacy = {
        "queries": [
            {"family": "old-train", "split": "train"},
            {"family": "old-test", "split": "test"},
        ]
    }
    original = deepcopy(legacy)
    review = {
        "review": {"status": "reviewed"},
        "observations": {
            r["path"]: {"sha256": r["sha256"], "description": "Observed " + r["path"]} for r in rows
        },
        "intents": [
            {
                "id": "new",
                "positives": ["0"],
                "translations": {"en": "New", "fr": "Nouveau", "ar": "جديد"},
            }
        ],
    }
    result = assemble({"images": rows}, legacy, review)
    assert legacy == original
    assert [q["study_role"] for q in result["queries"][:2]] == ["train", "diagnostic"]
    for query in result["queries"][2:]:
        assert {r["image_sha256"]: r["grade"] for r in query["judgments"]} == {"0": 2, "2": 0}
    review["observations"]["0"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="exact inspected"):
        assemble({"images": rows}, legacy, review)


def test_hybrid50_preserves_membership_and_tail_while_union_can_demote(tmp_path):
    artifact = {
        "version": 2,
        "kind": "logistic_regression",
        "feature_schema": 1,
        "candidate_budget": 50,
        "features": list(NAMES),
        "mean": [0] * len(NAMES),
        "scale": [1] * len(NAMES),
        "weights": [-1] + [0] * (len(NAMES) - 1),
        "intercept": 0,
        "compatibility": {"model_registry": {}, "inference_sha256": "fixture"},
        "feature_profile": "full",
        "rerank_scope": "hybrid50",
    }
    path = tmp_path / "model.json"
    save_json(path, artifact)
    rows = [candidate(i) for i in range(70)]
    model = load(path)
    assert model.rank(list(reversed(rows))) == [
        row["image_id"] for row in rows[:50][::-1] + rows[50:]
    ]
    assert set(model.rank(rows)[:50]) == {row["image_id"] for row in rows[:50]}
    artifact["rerank_scope"] = "union"
    save_json(path, artifact)
    assert load(path).rank(rows).index("000") == 69
    artifact["version"] = 1
    del artifact["rerank_scope"]
    del artifact["feature_profile"]
    save_json(path, artifact)
    assert load(path).scope == "union" and load(path).rank(rows).index("000") == 69


@pytest.mark.parametrize("profile", list(EXCLUDED))
def test_ablation_has_neutral_disabled_parameters_and_roundtrips(tmp_path, profile):
    rows = [
        {
            "query_id": "q",
            "query_family": "f",
            "query_split": "train",
            "image_split": "train",
            "training_eligible": True,
            "label": int(i < 2),
            "features": candidate(i),
        }
        for i in range(4)
    ]
    model = fit(
        rows,
        0.1,
        {"model_registry": {}, "inference_sha256": "fixture"},
        profile=profile,
        scope="hybrid50",
    )
    path = tmp_path / "model.json"
    save_json(path, model)
    assert load(path).profile == profile
    for name in EXCLUDED[profile]:
        i = NAMES.index(name)
        assert model["mean"][i] == model["weights"][i] == 0 and model["scale"][i] == 1
        invalid = deepcopy(model)
        invalid["weights"][i] = 0.01
        save_json(path, invalid)
        with pytest.raises(ValueError, match="Excluded features"):
            load(path)


def metrics(recall=1.0, mrr=0.5, ndcg=0.5):
    return {"candidate_recall_at_50": recall, "mrr_at_10": mrr, "ndcg_at_10": ndcg}


def test_validation_guard_catches_language_and_kind_losses_hidden_by_average():
    baseline = {"all": metrics(), "language:fr": metrics(), "kind:documents": metrics()}
    candidate_metrics = {key: metrics(mrr=0.6, ndcg=0.6) for key in baseline}
    spec = {
        "max_recall_at_50_drop": 0.0,
        "min_mrr_at_10_delta": 0.0,
        "min_ndcg_at_10_delta_exclusive": 0.0,
        "tolerance": 1e-12,
    }
    assert guard(candidate_metrics, baseline, spec) == []
    candidate_metrics["language:fr"]["candidate_recall_at_50"] = 0.9
    candidate_metrics["kind:documents"]["candidate_recall_at_50"] = 0.9
    assert set(guard(candidate_metrics, baseline, spec)) == {
        "recall:language:fr",
        "recall:kind:documents",
    }
    assert guard(baseline, baseline, spec) == ["ndcg_at_10"]


def test_stable_selection_uses_validation_then_stronger_l2():
    trials = [
        {"profile": "core", "scope": "hybrid50", "l2": l2, "validation": {"all": metrics()}}
        for l2 in (0.01, 0.1, 1)
    ]
    assert best(trials)["l2"] == 1
    trials[0]["validation"]["all"]["ndcg_at_10"] = 0.8
    assert best(trials)["l2"] == 0.01


def test_expansion_copy_does_not_change_legacy_signatures(tmp_path):
    source = tmp_path / "original.png"
    Image.new("RGB", (128, 96), "red").save(source)
    before = signature(source)
    target = tmp_path / "new" / "copy.png"
    copy_verified(source, target, digest(source))
    assert signature(source) == before
    assert source.stat().st_ino != target.stat().st_ino
    assert digest(source) == digest(target)
    Image.new("RGB", (128, 96), "blue").save(target)
    with pytest.raises(ValueError, match="differs"):
        copy_verified(source, target, digest(source))


def test_duplicate_guard_preserves_color_and_recognizes_resized_copies(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (64, 64), "red").save(first)
    Image.new("RGB", (128, 128), "red").save(second)
    assert near_duplicate(preview(first), preview(second))
    Image.new("RGB", (64, 64), (0, 130, 0)).save(second)
    assert not near_duplicate(preview(first), preview(second))


def test_protocol_profiles_match_implemented_ablation_allowlist():
    path = Path(__file__).resolve().parents[1] / "benchmarks/sources/retrieval-v2/protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["feature_profiles"] == {key: list(value) for key, value in EXCLUDED.items()}
    assert (
        len(protocol["feature_profiles"])
        * len(protocol["rerank_scopes"])
        * len(protocol["l2_grid"])
        == 40
    )


def test_study_runner_freezes_champions_and_keeps_hybrid_when_no_gain(tmp_path):
    from types import SimpleNamespace
    from benchmarks.experiments.reranker_study import run

    root = Path(__file__).resolve().parents[1]
    protocol = root / "benchmarks/sources/retrieval-v2/protocol.json"
    images, queries, records = [], [], []
    for fold in ("train", "validation", "test"):
        for kind in ("documents", "screenshots", "photos"):
            positive, negative = fold + "-" + kind + "-p", fold + "-" + kind + "-n"
            for key in (positive, negative):
                images.append(
                    {
                        "id": key,
                        "path": key,
                        "sha256": key,
                        "pixel_sha256": key,
                        "source_document": key,
                        "duplicate_group": key,
                        "split": fold,
                        "category": kind,
                    }
                )
            for lang in ("en", "fr", "ar"):
                query = {
                    "id": fold + "-" + kind + "-" + lang,
                    "family": fold + "-" + kind,
                    "text": fold + " " + kind + " " + lang,
                    "language": lang,
                    "kind": kind,
                    "split": fold,
                    "study_role": fold,
                    "source_exposure": "fresh-v2",
                    "review": {
                        "status": "reviewed",
                        "reviewer": "fixture",
                        "method": "fixture",
                        "date": "2026-10-20",
                    },
                    "judgments": [
                        {"image_sha256": positive, "grade": 2, "reason": "match"},
                        {"image_sha256": negative, "grade": 0, "reason": "distinct"},
                    ],
                }
                queries.append(query)
                a, b = candidate(0), candidate(1)
                a["image_id"], b["image_id"] = positive, negative
                records.append(
                    {
                        "query_id": query["id"],
                        "query": query,
                        "status": "ok",
                        "quality_deferred": fold == "test",
                        "trace": {
                            "generation": "g",
                            "searchable_images": 18,
                            "candidates": [a, b],
                            "rankings": {"hybrid": [positive, negative]},
                        },
                    }
                )
    corpus, judgments = tmp_path / "corpus.json", tmp_path / "judgments.json"
    save_json(corpus, {"images": images, "protocol_sha256": digest(protocol)})
    save_json(
        judgments,
        {
            "version": 1,
            "corpus_sha256": digest(corpus),
            "protocol_sha256": digest(protocol),
            "queries": queries,
        },
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    save_json(
        baseline / "config.json",
        {
            "corpus_sha256": digest(corpus),
            "judgments_sha256": digest(judgments),
            "candidate_feature_schema": 1,
            "search_limit": 50,
            "defer_test_metrics": True,
            "snapshot": {
                "generation": "g",
                "pipeline_spec": {"model_registry": {}, "inference_sha256": "fixture"},
            },
        },
    )
    (baseline / "queries.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    args = SimpleNamespace(
        corpus=corpus,
        judgments=judgments,
        protocol=protocol,
        baseline_dir=baseline,
        output=tmp_path / "out",
        legacy_model=None,
    )
    result = run(args)
    assert result["queries"] == 27 and result["trials"] == 40
    assert result["selection"] is None and result["guarded_trials"] == 0
    assert not (args.output / "model.json").exists()
    assert (args.output / "champion-union.json").exists()
    assert (args.output / "champion-hybrid50.json").exists()
    assert (
        len((args.output / "validation-trials.jsonl").read_text(encoding="utf-8").splitlines())
        == 360
    )
    with pytest.raises(ValueError, match="new output directory"):
        run(args)
    config_path = baseline / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["defer_test_metrics"] = False
    save_json(config_path, config)
    args.output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="deferred until validation"):
        run(args)
    assert not args.output.exists()
