"""Persistent indexing, publication and search without downloading model weights."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from realopen_lens.storage.catalog import Catalog
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.indexing.service import collection_status, run_index
from realopen_lens.indexing import service as indexing, processing
from realopen_lens.retrieval import search


@pytest.fixture
def app(tmp_path, monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    store = tmp_path / "index"
    calls = []
    monkeypatch.setattr(indexing, "verify", lambda *a: None)
    monkeypatch.setattr("realopen_lens.ocr.tesseract.shutil.which", lambda name: "tesseract")
    monkeypatch.setattr(
        "realopen_lens.ocr.tesseract._environment", lambda *a: {"version": "fixture"}
    )

    def ocr(path, *args):
        with Image.open(path) as im:
            text = "red error ERR_42" if im.getpixel((0, 0))[0] else "green garden"
        calls.append(("ocr", path.name))
        return {"text": text, "words": [], "warnings": [], "alternate_texts": []}

    monkeypatch.setattr("realopen_lens.ocr.tesseract.extract", ocr)

    class Encoder:
        def __init__(self, kind):
            self.kind = kind

        def passages(self, text):
            return [text] if text else []

        def encode(self, inputs, **kw):
            values = []
            for item in inputs:
                if isinstance(item, Path):
                    calls.append(("vision", item.name))
                    with Image.open(item) as im:
                        red = bool(im.getpixel((0, 0))[0])
                else:
                    calls.append(("semantic", item))
                    red = "red" in item
                value = np.zeros(768 if self.kind == "vision" else 384, dtype=np.float32)
                value[0 if red else 1] = 1
                values.append(value)
            return np.stack(values)

    monkeypatch.setattr(processing, "encoder", lambda kind, config: Encoder(kind))

    def run(**extra):
        options = {
            "root": str(root),
            "store": str(store),
            "collection": "personal",
            "models": str(tmp_path / "models"),
            "threads": 2,
            "languages": "eng",
            "rehash": False,
            **extra,
        }

        def stage(catalog, job, kind, command, **kw):
            processing.process_stage(catalog, job, kind)

        return run_index(options, stage_runner=stage)

    return root, store, calls, run


def image(path, color="red"):
    Image.new("RGB", (64, 48), color).save(path)


def test_multilingual_manifest_and_search_ignore_windows_default_encoding(app, monkeypatch):
    root, store, calls, run = app
    image(root / "صورة.png")
    from realopen_lens.ocr import tesseract

    extract = tesseract.extract

    def multilingual(*args, **kwargs):
        result = extract(*args, **kwargs)
        result["text"] += " العربية"
        return result

    monkeypatch.setattr(tesseract, "extract", multilingual)
    run()
    read_text = Path.read_text

    def windows_default(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", windows_default)
    catalog = Catalog(store, readonly=True)
    try:
        _, generation = catalog.snapshot("personal")
        manifest = load_manifest(catalog, generation)
        assert manifest["images"][0]["paths"] == ["صورة.png"]
        assert "العربية" in manifest["ocr"][0]["text"]
    finally:
        catalog.close()
    result = search.search_collection("العربية", index_dir=store, mode="lexical")
    assert len(result) == 1 and result[0].paths == (str(root / "صورة.png"),)


def test_roundtrip_duplicates_restart_and_lexical_search_without_models(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    (root / "copy.png").write_bytes((root / "one.png").read_bytes())
    image(root / "two.png", "green")
    result = run()
    assert result.files == 3 and result.images == 2 and result.failed_files == 0
    assert len([c for c in calls if c[0] == "vision"]) == 2
    found = search.search_collection("ERR_42", index_dir=store, mode="lexical")
    assert len(found) == 1 and len(found[0].paths) == 2
    assert Path(found[0].thumbnail).exists()
    calls.clear()
    again = run()
    assert again.generation == result.generation and again.reused_images == 2 and calls == []
    assert collection_status("personal", store)["job"]["status"] == "completed"


def test_modified_moved_removed_and_bad_files_are_not_stale(app):
    root, store, calls, run = app
    image(root / "one.png")
    image(root / "two.png", "green")
    run()
    old_id = search.search_collection("ERR_42", index_dir=store, mode="lexical")[0].image_id
    (root / "one.png").rename(root / "renamed.png")
    calls.clear()
    moved = run()
    assert moved.removed_files == 1 and not calls
    assert search.search_collection("ERR_42", index_dir=store, mode="lexical")[0].image_id == old_id
    image(root / "renamed.png", "green")
    (root / "two.png").unlink()
    (root / "broken.png").write_bytes(b"not an image")
    changed = run()
    assert changed.failed_files == 1 and changed.images == 1 and changed.removed_files == 1
    assert search.search_collection("ERR_42", index_dir=store, mode="lexical") == []
    assert collection_status("personal", store)["job"]["failures"][0]["path"] == "broken.png"
    (root / "broken.png").unlink()
    (root / "renamed.png").unlink()
    assert run().images == 0
    assert search.search_collection("anything", index_dir=store, mode="lexical") == []


def test_failed_publication_keeps_previous_generation(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    first = run()
    image(root / "two.png", "green")

    def fail(*a):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(indexing, "publish", fail)
    with pytest.raises(OSError, match="disk failure"):
        run()
    catalog = Catalog(store, readonly=True)
    assert catalog.collection("personal")["active_generation"] == first.generation
    catalog.close()
    assert len(search.search_collection("ERR_42", index_dir=store, mode="lexical")) == 1
    assert collection_status("personal", store)["job"]["status"] == "failed"


def test_cache_and_generation_corruption_rebuild_and_pipeline_versions(app):
    root, store, calls, run = app
    image(root / "one.png")
    first = run()
    catalog = Catalog(store)
    _, generation = catalog.snapshot("personal")
    manifest = load_manifest(catalog, generation)
    (store / manifest["artifacts"]["vision"]["path"]).write_bytes(b"corrupt")
    catalog.close()
    calls.clear()
    assert run().generation != first.generation and not calls
    catalog = Catalog(store)
    record = catalog.db.execute("SELECT artifact FROM analyses WHERE stage='vision'").fetchone()
    (store / record[0]).unlink()
    catalog.close()
    run()
    assert len([c for c in calls if c[0] == "vision"]) == 1
    calls.clear()
    run(languages="fra")
    assert len([c for c in calls if c[0] == "vision"]) == 1


def test_source_mutation_during_processing_is_an_error(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    original = processing.metadata

    def changed(path):
        result = original(path)
        image(path, "green")
        return result

    monkeypatch.setattr(processing, "metadata", changed)
    result = run()
    assert result.failed_files == 1 and result.images == 0
    assert "changed" in collection_status("personal", store)["job"]["failures"][0]["error"].lower()


def test_index_inside_root_is_excluded_and_collection_root_is_fixed(app):
    root, store, calls, run = app
    image(root / "one.png")
    nested = root / "private-index"
    run(store=str(nested))
    assert run(store=str(nested)).files == 1
    other = root.parent / "different"
    other.mkdir()
    with pytest.raises(ValueError, match="different root"):
        run(root=str(other), store=str(nested))


def test_hybrid_uses_existing_published_vectors(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    image(root / "two.png", "green")
    run()
    monkeypatch.setattr(search, "verify", lambda *a: None)

    def query(text, kind, models, threads):
        value = np.zeros(768 if kind == "siglip2" else 384, dtype=np.float32)
        value[0] = 1
        return value

    monkeypatch.setattr(search, "query_vector", query)
    result = search.search_collection("red ERR_42", index_dir=store)
    assert result[0].paths == (str(root / "one.png"),)
    assert result[0].visual_similarity == pytest.approx(1)


def test_search_filters_changed_and_missing_sources_before_refresh(app):
    root, store, calls, run = app
    image(root / "one.png")
    (root / "copy.png").write_bytes((root / "one.png").read_bytes())
    run()
    (root / "copy.png").unlink()
    found = search.search_collection("ERR_42", index_dir=store, mode="lexical")
    assert found[0].paths == (str(root / "one.png"),)
    image(root / "one.png", "green")
    assert search.search_collection("ERR_42", index_dir=store, mode="lexical") == []


def test_publication_failure_after_artifact_rename_preserves_snapshot(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    original = run()
    image(root / "two.png", "green")

    def fail(*args):
        raise OSError("catalog unavailable")

    monkeypatch.setattr(Catalog, "publish", fail)
    with pytest.raises(OSError, match="catalog unavailable"):
        run()
    assert collection_status("personal", store)["generation"] == original.generation
    assert len(search.search_collection("ERR_42", index_dir=store, mode="lexical")) == 1


def test_interrupted_stage_resumes_completed_work(app):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    image(root / "two.png", "green")
    catalog = Catalog(store)
    previous = catalog.db.execute("SELECT * FROM jobs LIMIT 1").fetchone()
    import json

    stale = catalog.create_job(previous["collection_id"], json.loads(previous["config"]))
    catalog.close()
    calls.clear()
    assert run().reused_images == 1
    assert len([c for c in calls if c[0] == "vision"]) == 1
    catalog = Catalog(store, readonly=True)
    assert catalog.job(stale)["status"] == "interrupted"
    catalog.close()


def test_corrupt_thumbnail_is_hidden_and_repaired(app):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    thumbnail = Path(
        search.search_collection("ERR_42", index_dir=store, mode="lexical")[0].thumbnail
    )
    thumbnail.write_bytes(b"broken")
    assert search.search_collection("ERR_42", index_dir=store, mode="lexical")[0].thumbnail is None
    calls.clear()
    run()
    assert not calls
    assert search.search_collection("ERR_42", index_dir=store, mode="lexical")[0].thumbnail


def test_unreadable_subtree_never_publishes_false_deletions(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    first = run()

    def unreadable(*args, **kwargs):
        kwargs["onerror"](PermissionError("Cannot scan subtree"))

    monkeypatch.setattr(indexing.os, "walk", unreadable)
    with pytest.raises(PermissionError, match="Cannot scan"):
        run()
    assert collection_status("personal", store)["generation"] == first.generation


def test_partial_failure_has_nonzero_cli_exit(app, monkeypatch):
    from typer.testing import CliRunner
    from realopen_lens.cli import app as cli

    root, store, calls, run = app
    image(root / "one.png")
    (root / "broken.png").write_bytes(b"broken")
    result = run()
    options = []

    def index(*args, **kwargs):
        options.append(kwargs)
        return result

    monkeypatch.setattr(indexing, "index_collection", index)
    runner = CliRunner()
    response = runner.invoke(cli, ["index", str(root), "--log"])
    assert response.exit_code == 1
    assert '"completed_with_errors"' in response.stdout
    assert options[0]["log"] is True
    status = runner.invoke(cli, ["status", "--index-dir", str(store)])
    assert status.exit_code == 0 and "broken.png" in status.stdout
    found = runner.invoke(cli, ["search", "ERR_42", "--mode", "lexical", "--index-dir", str(store)])
    assert found.exit_code == 0 and "one.png" in found.stdout


def test_special_files_do_not_block_scanning(app):
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("Named pipes are a POSIX filesystem feature")
    root, store, calls, run = app
    image(root / "one.png")
    os.mkfifo(root / "pipe.png")
    (root / "loop.png").symlink_to("loop.png")
    result = run()
    assert result.failed_files == 2 and result.images == 1
    assert len(search.search_collection("ERR_42", index_dir=store, mode="lexical")) == 1


def test_empty_ocr_has_no_semantic_rows_but_remains_visually_searchable(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    monkeypatch.setattr("realopen_lens.ocr.tesseract.extract", lambda *a: {"text": ""})
    run()
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(
        search,
        "query_vector",
        lambda text, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0],
    )
    assert search.search_collection("a photo", index_dir=store, mode="semantic") == []
    assert len(search.search_collection("a photo", index_dir=store, mode="visual")) == 1


def test_logging_reports_cache_reuse_without_invalidating_it(app, caplog):
    import logging

    root, store, calls, run = app
    image(root / "one.png")
    first = run()
    calls.clear()
    with caplog.at_level(logging.INFO, logger="realopen_lens"):
        second = run(log=True)
    assert second.generation == first.generation and not calls
    assert "1 fully cached images" in caplog.text
    assert "[ocr] No pending images" in caplog.text
    assert "Keeping published generation" in caplog.text


def test_ocr_failure_keeps_visual_results_and_retry_reuses_completed_stages(app, monkeypatch):
    import json
    import subprocess
    from realopen_lens.ocr import tesseract as ocr

    root, store, calls, run = app
    image(root / "one.png")
    (root / "copy.png").write_bytes((root / "one.png").read_bytes())
    image(root / "two.png", "green")
    original = ocr.extract
    attempts = []

    def fail(path, *args):
        with Image.open(path) as source:
            red = bool(source.getpixel((0, 0))[0])
        if red:
            attempts.append(path)
            raise subprocess.TimeoutExpired("tesseract", 15)
        return original(path, *args)

    monkeypatch.setattr("realopen_lens.ocr.tesseract.extract", fail)
    first = run()
    assert first.images == 2 and first.failed_files == 0 and first.degraded_files == 2
    assert first.degraded_images == 1 and first.status == "completed_with_errors"
    assert len(attempts) == 1  # Duplicate paths share one failed OCR attempt.
    catalog = Catalog(store, readonly=True)
    before = [
        dict(r)
        for r in catalog.db.execute(
            "SELECT * FROM analyses WHERE stage IN ('vision','metadata') ORDER BY image_id,stage"
        )
    ]
    _, generation = catalog.snapshot("personal")
    manifest = load_manifest(catalog, generation)
    red_id = next(r["image_id"] for r in manifest["images"] if r["status"] == "degraded")
    assert red_id not in manifest["passage_ids"]
    assert (
        json.loads(
            catalog.analysis(
                json.loads(catalog.job(first.job_id)["config"])["pipeline"], red_id, "semantic"
            )["payload"]
        )["status"]
        == "skipped"
    )
    catalog.close()
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(
        search,
        "query_vector",
        lambda text, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0],
    )
    found = search.search_collection("red photo", index_dir=store, mode="visual")
    assert found[0].image_id == red_id and len(found[0].paths) == 2
    assert found[0].status == "degraded" and found[0].issues[0]["stage"] == "ocr"
    assert found[0].issues[0]["error_type"] == "TimeoutExpired"
    assert search.search_collection("ERR_42", index_dir=store, mode="lexical") == []
    status = collection_status("personal", store)["job"]
    assert status["degraded"] == 2 and status["searchable"] == 3 and not status["failures"]
    calls.clear()
    second = run()
    assert second.generation != first.generation and len(attempts) == 2
    assert second.reused_stages == {"metadata": 2, "ocr": 1, "vision": 2, "semantic": 1}
    assert calls == []  # Healthy OCR and both visual/metadata records are reused.
    monkeypatch.setattr("realopen_lens.ocr.tesseract.extract", original)
    recovered = run()
    assert recovered.status == "completed" and recovered.degraded_files == 0
    assert not [c for c in calls if c[0] == "vision"]
    assert len(search.search_collection("ERR_42", index_dir=store, mode="lexical")) == 1
    catalog = Catalog(store, readonly=True)
    after = [
        dict(r)
        for r in catalog.db.execute(
            "SELECT * FROM analyses WHERE stage IN ('vision','metadata') ORDER BY image_id,stage"
        )
    ]
    assert after == before
    catalog.close()


def test_retried_ocr_invalidates_previous_semantic_embeddings(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    catalog = Catalog(store)
    with catalog.db:
        catalog.db.execute("DELETE FROM analyses WHERE stage='ocr'")
    catalog.close()
    monkeypatch.setattr(
        "realopen_lens.ocr.tesseract.extract", lambda *a: {"text": "new transcription"}
    )
    calls.clear()
    result = run()
    assert result.reused_stages["vision"] == 1 and result.reused_stages["semantic"] == 0
    assert calls == [("semantic", "new transcription")]
    assert search.search_collection("ERR_42", index_dir=store, mode="lexical") == []
    assert len(search.search_collection("transcription", index_dir=store, mode="lexical")) == 1


def test_semantic_failure_preserves_ocr_and_drops_stale_vectors(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    catalog = Catalog(store)
    record = catalog.db.execute("SELECT artifact FROM analyses WHERE stage='semantic'").fetchone()
    (store / record[0]).unlink()
    catalog.close()
    original = processing.encoder

    def broken(stage, config):
        model = original(stage, config)
        if stage == "semantic":

            def fail(*args, **kwargs):
                raise RuntimeError("semantic inference failed")

            model.encode = fail
        return model

    monkeypatch.setattr(processing, "encoder", broken)
    calls.clear()
    result = run()
    assert result.images == 1 and result.failed_files == 0 and result.degraded_files == 1
    assert calls == []
    found = search.search_collection("ERR_42", index_dir=store, mode="lexical")
    assert found[0].text == "red error ERR_42" and found[0].issues[0]["stage"] == "semantic"
    catalog = Catalog(store, readonly=True)
    _, generation = catalog.snapshot("personal")
    assert load_manifest(catalog, generation)["passage_ids"] == []
    catalog.close()
    monkeypatch.setattr(processing, "encoder", original)
    assert run().degraded_files == 0
    assert all(c[0] == "semantic" for c in calls)


def test_source_mutation_during_failed_ocr_remains_a_hard_failure(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")

    def fail(path, *args):
        image(path, "green")
        raise RuntimeError("OCR failed after file changed")

    monkeypatch.setattr("realopen_lens.ocr.tesseract.extract", fail)
    result = run()
    assert result.failed_files == 1 and result.images == 0 and result.degraded_files == 0


def test_degraded_only_run_has_clean_json_and_nonzero_cli_exit(app, monkeypatch):
    from typer.testing import CliRunner
    from realopen_lens.cli import app as cli
    import json

    root, store, calls, run = app
    image(root / "one.png")

    def fail(*args):
        raise RuntimeError("OCR unavailable")

    monkeypatch.setattr("realopen_lens.ocr.tesseract.extract", fail)
    result = run()
    monkeypatch.setattr(indexing, "index_collection", lambda *a, **kw: result)
    response = CliRunner().invoke(cli, ["index", str(root)])
    assert response.exit_code == 1
    output = json.loads(response.stdout)
    assert output["failed_files"] == 0 and output["degraded_files"] == 1


def test_candidate_trace_uses_production_scores_without_changing_results(app, monkeypatch):
    from benchmarks.runtime.collections import snapshot

    root, store, calls, run = app
    image(root / "one.png")
    image(root / "two.png", "green")
    run()
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(
        search,
        "query_vector",
        lambda text, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0],
    )
    trace = {"stale": True}
    normal = search.search_collection("red ERR_42", index_dir=store, limit=50)
    traced = search.search_collection("red ERR_42", index_dir=store, limit=50, trace=trace)
    assert normal == traced and "stale" not in trace
    assert [r.image_id for r in traced] == trace["rankings"]["hybrid"]
    assert trace["searchable_images"] == 2
    candidate = next(r for r in trace["candidates"] if r["image_id"] == traced[0].image_id)
    assert candidate["visual_cosine"] == traced[0].visual_similarity
    assert candidate["lexical_bm25"] < 0 and candidate["channel_ranks"]["lexical"] == 1
    assert candidate["semantic_cosine"] == pytest.approx(1)
    assert candidate["hybrid_score"] > candidate["visual_cosine"]
    assert candidate["width"] == 64 and candidate["ocr_characters"] > 0
    corpus = {
        "images": [
            {"sha256": r.image_id, "path": Path(r.paths[0]).relative_to(root).as_posix()}
            for r in traced
        ]
    }
    state = snapshot(store, "personal", corpus)
    assert state["unique_images"] == 2 and state["paths"] == 2
    with pytest.raises(ValueError, match="exactly"):
        snapshot(store, "personal", {"images": corpus["images"][:1]})
    image(root / "one.png", "blue")
    with pytest.raises(ValueError, match="Source changed"):
        snapshot(store, "personal", corpus)
