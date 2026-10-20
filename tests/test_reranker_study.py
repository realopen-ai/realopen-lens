"""Focused regression coverage for experimental reranking."""

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
from realopen_lens.indexing.service import signature
from realopen_lens.learning.features import NAMES
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
