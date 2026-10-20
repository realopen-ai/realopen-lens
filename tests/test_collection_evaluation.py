"""Production candidate tracing, reviewed labels, and evaluation resume contracts."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.datasets.scale import group_corpus
from benchmarks.experiments import collections
from benchmarks.metrics.collection import evaluate
from benchmarks.reporting.collections import candidate_rows
from realopen_lens.retrieval.ranking import hybrid_rank


@pytest.fixture(autouse=True)
def mock_model_integrity_for_unit_tests(monkeypatch):
    monkeypatch.setattr(collections, "verify", lambda *args: None)


@pytest.fixture(params=["utf-8", "cp1252"])
def default_text_encoding(request, monkeypatch):
    original = Path.read_text

    def read_text(path, encoding=None, errors=None):
        return original(path, encoding=encoding or request.param, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)


def corpus_files(tmp_path):
    rows = [
        {
            "id": key,
            "path": key,
            "sha256": key,
            "pixel_sha256": key,
            "source_document": key,
            "duplicate_group": key,
            "split": split,
            "category": "photos",
        }
        for key, split in [("a", "train"), ("b", "train"), ("c", "test")]
    ]
    corpus = tmp_path / "corpus.json"
    save_json(corpus, {"images": rows})
    query = {
        "id": "q",
        "text": "a red car",
        "language": "en",
        "family": "car",
        "split": "train",
        "review": {
            "status": "reviewed",
            "reviewer": "fixture",
            "method": "fixture",
            "date": "2026-10-20",
        },
        "judgments": [
            {"image_sha256": "a", "grade": 2, "reason": "red car"},
            {"image_sha256": "b", "grade": 0, "reason": "blue bicycle"},
        ],
    }
    bundle = {"version": 1, "corpus_sha256": digest(corpus), "queries": [query]}
    path = tmp_path / "judgments.json"
    save_json(path, bundle)
    return corpus, path, rows, bundle


@pytest.mark.parametrize("field", ["source_document", "duplicate_group", "pixel_sha256"])
def test_leakage_across_document_groups_is_rejected(tmp_path, field):
    corpus, path, rows, bundle = corpus_files(tmp_path)
    rows[-1][field] = rows[0][field]
    save_json(corpus, {"images": rows})
    bundle["corpus_sha256"] = digest(corpus)
    save_json(path, bundle)
    with pytest.raises(ValueError, match="Split leakage"):
        load(corpus, path)


def test_query_families_and_review_are_required(tmp_path):
    corpus, path, rows, bundle = corpus_files(tmp_path)
    assert load(corpus, path)[1] == bundle
    other = deepcopy(bundle["queries"][0])
    other.update(id="q-fr", text="une voiture rouge", language="fr", split="test")
    bundle["queries"].append(other)
    save_json(path, bundle)
    with pytest.raises(ValueError, match="query_family"):
        load(corpus, path)
    bundle["queries"] = [bundle["queries"][0]]
    bundle["queries"][0]["review"]["status"] = "draft"
    save_json(path, bundle)
    with pytest.raises(ValueError, match="review provenance"):
        load(corpus, path)


def test_known_grades_and_unjudged_candidates_are_distinct(tmp_path):
    assert evaluate(["x", "a", "b"], {"a": 2, "b": 1, "n": 0})["mrr_at_10"] == 0.5
    assert evaluate(["a", "b"], {"a": 2, "b": 1})["ndcg_at_10"] == pytest.approx(1)
    assert evaluate(["a"], {"a": 2, "b": 1})["candidate_recall_at_50"] == 0.5
    with pytest.raises(ValueError, match="repeats"):
        evaluate(["a", "a"], {"a": 2})
    _, _, images, bundle = corpus_files(tmp_path)
    record = {
        "trace": {
            "candidates": [{"image_id": key, "visual_cosine": 0.1} for key in ("a", "b", "c")]
        }
    }
    rows = candidate_rows(record, bundle["queries"][0], {r["sha256"]: r for r in images})
    assert [r["label"] for r in rows] == [1, 0, None]
    assert [r["training_eligible"] for r in rows] == [True, True, False]
    assert "label" not in rows[0]["features"]


def test_candidate_trace_preserves_ranking_and_passage_aggregation():
    rng = np.random.default_rng(42)
    visual = rng.normal(size=(80, 8)).astype(np.float32)
    semantic = rng.normal(size=(90, 4)).astype(np.float32)
    ids = [str(i) for i in range(80)]
    passage_ids = [str(i % 70) for i in range(90)]
    args = (
        visual,
        np.ones(8),
        ids,
        semantic,
        np.ones(4),
        passage_ids,
        "ERR_42",
        [{"image_id": key, "text": "ERR_42" if key == "1" else ""} for key in ids],
    )
    trace = {}
    assert hybrid_rank(*args, trace=trace) == hybrid_rank(*args)
    candidates = trace["candidates"]
    assert len({r["image_id"] for r in candidates}) == len(candidates)
    assert [r["image_id"] for r in candidates[:50]] == trace["rankings"]["hybrid"]
    assert all(np.isfinite(r["hybrid_score"]) for r in candidates)


def test_grouping_is_transitive_and_family_safe():
    rows = [
        {"sha256": key, "dhash": dhash, "source_document": family}
        for key, dhash, family in [
            ("a", "0000000000000000", "doc"),
            ("b", "ffffffffffffffff", "doc"),
            ("c", "fffffffffffffffe", "other"),
        ]
    ]
    group_corpus(rows)
    assert len({r["duplicate_group"] for r in rows}) == 1
    assert len({r["split"] for r in rows}) == 1


@pytest.mark.parametrize("use_reranker", [False, True])
@pytest.mark.parametrize("defer_test", [False, True])
def test_evaluation_resumes_records_and_rejects_identity_changes(
    tmp_path, monkeypatch, default_text_encoding, use_reranker, defer_test
):
    corpus, path, rows, bundle = corpus_files(tmp_path)
    if defer_test:
        for row in rows:
            row["split"] = "test"
        save_json(corpus, {"images": rows})
        bundle["corpus_sha256"] = digest(corpus)
        bundle["queries"][0].update(split="test", study_role="test")
        save_json(path, bundle)

        def forbidden(*args):
            raise AssertionError("Fresh test quality must not be scored before model freeze")

        monkeypatch.setattr(collections, "evaluate", forbidden)
    state = {"generation": "g", "unique_images": 3, "paths": 3}
    monkeypatch.setattr(collections, "snapshot", lambda *a, **kw: state)
    calls = []
    reranker = tmp_path / "model.json"
    reranker.write_text("{}", encoding="utf-8")

    def search(*a, trace, **kw):
        calls.append(1)
        trace.update(
            generation="g",
            searchable_images=3,
            rankings={
                mode: ["a", "b", "c"] for mode in ("visual", "semantic", "lexical", "hybrid")
            },
            candidates=[{"image_id": key, "visual_cosine": 0.1} for key in ("a", "b", "c")],
        )
        if use_reranker:
            assert kw["reranker"] == reranker
            trace["rankings"]["logistic"] = ["c", "b", "a"]
            trace["reranker_sha256"] = digest(reranker)
            return [SimpleNamespace(image_id=key) for key in ("c", "b", "a")]
        return [SimpleNamespace(image_id=key) for key in ("a", "b", "c")]

    monkeypatch.setattr(collections, "search_collection", search)
    args = SimpleNamespace(
        corpus=corpus,
        judgments=path,
        index_dir=tmp_path / "index",
        collection="اختبار",
        models=Path("models"),
        threads=2,
        output=tmp_path / "out",
        retry_errors=False,
        reranker=reranker if use_reranker else None,
        defer_test_metrics=defer_test,
    )
    collections.run(args)
    with sqlite3.connect(args.output / "results.sqlite") as db:
        before = db.execute("SELECT * FROM queries").fetchall()
    collections.run(args)
    with sqlite3.connect(args.output / "results.sqlite") as db:
        assert db.execute("SELECT * FROM queries").fetchall() == before
    assert len(calls) == 1
    args.visual_candidates = 200
    with pytest.raises(ValueError, match="identity changed|require 50"):
        collections.run(args)
    args.visual_candidates = 50
    exported = [
        json.loads(line)
        for line in (args.output / f"training-{'test' if defer_test else 'train'}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(exported) == 2 and all(r["label"] is not None for r in exported)
    if defer_test:
        record = json.loads(before[0][1])
        assert record["metrics"] == {} and record["quality_deferred"]
        assert "pool_recall" not in record
        summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
        assert summary["quality_deferred_queries"] == 1
        assert "hybrid/role:test" not in summary["aggregate"]
    if use_reranker:
        report = (args.output / "report.md").read_text(encoding="utf-8")
        summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
        assert ("| logistic |" in report) is not defer_test
        assert "reranked_end_to_end_seconds" in summary
        assert "hybrid_end_to_end_seconds" not in summary
        assert (args.output / "reranker.json").read_bytes() == reranker.read_bytes()
        reranker.write_text('{"changed":true}', encoding="utf-8")
        with pytest.raises(ValueError, match="identity changed"):
            collections.run(args)
        reranker.write_text("{}", encoding="utf-8")
    state["generation"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        collections.run(args)


def test_query_failures_are_retained_and_remaining_queries_run(tmp_path, monkeypatch):
    corpus, path, rows, bundle = corpus_files(tmp_path)
    other = deepcopy(bundle["queries"][0])
    other.update(id="q-fr", text="une voiture rouge", language="fr")
    bundle["queries"].append(other)
    save_json(path, bundle)
    state = {"generation": "g", "unique_images": 3, "paths": 3}
    monkeypatch.setattr(collections, "snapshot", lambda *a, **kw: state)
    calls = []

    def fail(*a, **kw):
        calls.append(1)
        raise TimeoutError("fixture")

    monkeypatch.setattr(collections, "search_collection", fail)
    args = SimpleNamespace(
        corpus=corpus,
        judgments=path,
        index_dir=tmp_path / "index",
        collection="test",
        models=Path("models"),
        threads=2,
        output=tmp_path / "out",
        retry_errors=False,
    )
    with pytest.raises(ValueError, match="2 queries failed"):
        collections.run(args)
    assert len(calls) == 2
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_queries"] == 2 and summary["execution_success_rate"] == 0
    assert len((args.output / "queries.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_explicit_cross_split_labels_and_invalid_grades_are_rejected(tmp_path):
    corpus, path, rows, bundle = corpus_files(tmp_path)
    bundle["queries"][0]["judgments"].append(
        {"image_sha256": "c", "grade": 0, "reason": "other split"}
    )
    save_json(path, bundle)
    with pytest.raises(ValueError, match="share a split"):
        load(corpus, path)
    bundle["queries"][0]["judgments"].pop()
    bundle["queries"][0]["judgments"][0]["grade"] = True
    save_json(path, bundle)
    with pytest.raises(ValueError, match="grade"):
        load(corpus, path)


def test_retry_retains_all_failed_attempts(tmp_path, monkeypatch):
    corpus, path, rows, bundle = corpus_files(tmp_path)
    monkeypatch.setattr(
        collections,
        "snapshot",
        lambda *a, **kw: {"generation": "g", "unique_images": 3, "paths": 3},
    )

    def fail(*a, **kw):
        raise TimeoutError("fixture")

    monkeypatch.setattr(collections, "search_collection", fail)
    args = SimpleNamespace(
        corpus=corpus,
        judgments=path,
        index_dir=tmp_path / "index",
        collection="test",
        models=Path("models"),
        threads=2,
        output=tmp_path / "out",
        retry_errors=False,
    )
    with pytest.raises(ValueError, match="1 queries failed"):
        collections.run(args)
    args.retry_errors = True
    with pytest.raises(ValueError, match="1 queries failed"):
        collections.run(args)
    with sqlite3.connect(args.output / "results.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
    assert len((args.output / "attempts.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_pixel_identity_ignores_encoding_but_preserves_shape(tmp_path):
    from PIL import Image
    from benchmarks.datasets.scale import pixel_hash

    Image.new("RGB", (8, 4), "red").save(tmp_path / "a.png")
    Image.new("RGB", (8, 4), "red").save(tmp_path / "b.bmp")
    Image.new("RGB", (4, 8), "red").save(tmp_path / "c.png")
    assert pixel_hash(tmp_path / "a.png") == pixel_hash(tmp_path / "b.bmp")
    assert pixel_hash(tmp_path / "a.png") != pixel_hash(tmp_path / "c.png")


def test_frozen_multilingual_judgments_and_corpus_are_consistent():
    base = Path(__file__).resolve().parents[1] / "benchmarks/sources/retrieval-v1"
    corpus, bundle = load(base / "corpus.json", base / "judgments.json")
    assert 1900 <= len(corpus["images"]) <= 2100
    assert len({r["pixel_sha256"] for r in corpus["images"]}) == len(corpus["images"])
    assert {q["split"] for q in bundle["queries"]} == {"train", "validation", "test"}
    families = {}
    for query in bundle["queries"]:
        families.setdefault(query["family"], []).append(query)
    for queries in families.values():
        assert {q["language"] for q in queries} == {"en", "fr", "ar"}
        assert all(q["judgments"] == queries[0]["judgments"] for q in queries)
        assert all(q["split"] == queries[0]["split"] for q in queries)
        assert all(q["review"]["independent_human_review"] is False for q in queries)


def test_frozen_references_preserve_bytes_with_windows_git_settings(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to exercise checkout line-ending conversion")
    root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    attributes = root / ".gitattributes"
    if attributes.exists():
        shutil.copyfile(attributes, checkout / ".gitattributes")
    originals = {}
    for source in (root / "benchmarks/sources/retrieval-v1").glob("*.json"):
        target = checkout / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        originals[target] = source.read_bytes()
        target.write_bytes(originals[target])

    def run_git(*args):
        subprocess.run(
            [
                git,
                "-c",
                "core.autocrlf=true",
                "-c",
                f"core.attributesFile={os.devnull}",
                "-C",
                str(checkout),
                *args,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

    run_git("init")
    run_git("add", "--force", "--all")
    for target in originals:
        target.unlink()
    run_git("checkout-index", "--all", "--force")
    for target, original in originals.items():
        assert target.read_bytes() == original, f"Checkout changed frozen bytes: {target.name}"
    base = checkout / "benchmarks/sources/retrieval-v1"
    load(base / "corpus.json", base / "judgments.json")


def test_pixel_identity_uses_production_orientation_and_alpha(tmp_path):
    from PIL import Image
    from benchmarks.datasets.scale import pixel_hash

    Image.new("RGBA", (4, 8), (255, 0, 0, 0)).save(tmp_path / "hidden-red.png")
    Image.new("RGB", (4, 8), "white").save(tmp_path / "white.bmp")
    assert pixel_hash(tmp_path / "hidden-red.png") == pixel_hash(tmp_path / "white.bmp")
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (8, 4), "white").save(tmp_path / "rotated.png", exif=exif)
    assert pixel_hash(tmp_path / "rotated.png") == pixel_hash(tmp_path / "white.bmp")


def test_reference_correction_is_auditable_and_preserves_unaffected_queries(default_text_encoding):
    base = Path(__file__).resolve().parents[1] / "benchmarks/sources/retrieval-v1"
    correction = json.loads((base / "corrections-v2.json").read_text(encoding="utf-8"))
    assert correction["initial_judgments_sha256"] == digest(base / "judgments.initial.json")
    assert correction["corrected_judgments_sha256"] == digest(base / "judgments.json")
    initial_bundle = json.loads((base / "judgments.initial.json").read_text(encoding="utf-8"))
    initial = {q["id"]: q for q in initial_bundle["queries"]}
    corrected = json.loads((base / "judgments.json").read_text(encoding="utf-8"))
    affected = set(correction["affected_queries"])
    assert len(affected) == 3 and corrected["revision"] == 2
    for query in corrected["queries"]:
        previous = initial[query["id"]]
        assert [(r["image_sha256"], r["grade"]) for r in query["judgments"]] == [
            (r["image_sha256"], r["grade"]) for r in previous["judgments"]
        ]
        if query["id"] in affected:
            assert query["text"] == correction["after"][query["language"]]
            assert previous["text"] == correction["before"][query["language"]]
            assert query["review"]["correction"]
        else:
            assert query["text"] == previous["text"]
