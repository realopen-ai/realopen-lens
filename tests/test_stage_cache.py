"""Incremental work, invalidation dependencies, durable reuse and live accounting."""

import json
import py_compile

from test_collections import app as _app, image
from realopen_lens.indexing import service, stage_cache
from realopen_lens.storage.catalog import Catalog

app = _app


def counts(calls):
    return {stage: sum(c[0] == stage for c in calls) for stage in ("ocr", "vision", "semantic")}


def test_adding_one_image_processes_only_the_new_content(app):
    root, store, calls, run = app
    image(root / "one.png")
    image(root / "two.png", "green")
    run()
    calls.clear()
    image(root / "three.png", "blue")
    result = run()
    assert counts(calls) == {"ocr": 1, "vision": 1, "semantic": 1}
    assert result.reused_stages == dict.fromkeys(("metadata", "ocr", "vision", "semantic"), 2)
    assert result.processed_stages == dict.fromkeys(("metadata", "ocr", "vision", "semantic"), 1)
    assert result.reused_images == 2
    status = service.collection_status("personal", store)["job"]
    assert status["phase_cached_images"] == 2 and status["phase_processed_images"] == 1


def test_ocr_environment_change_preserves_metadata_and_vision(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    calls.clear()
    monkeypatch.setattr(
        "realopen_lens.ocr.tesseract._environment", lambda *a: {"version": "changed engine"}
    )
    result = run()
    assert result.reused_stages == {"metadata": 1, "ocr": 0, "vision": 1, "semantic": 0}
    assert counts(calls) == {"ocr": 1, "vision": 0, "semantic": 1}
    calls.clear()
    unchanged = run()
    assert not calls and unchanged.reused_images == 1


def test_ocr_languages_only_invalidate_ocr_and_semantic(app):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    calls.clear()
    changed = run(languages="fra")
    assert changed.reused_stages == {"metadata": 1, "ocr": 0, "vision": 1, "semantic": 0}
    assert counts(calls) == {"ocr": 1, "vision": 0, "semantic": 1}
    calls.clear()
    returned = run(languages="eng")
    assert not calls and returned.reused_images == 1


def test_other_collection_precision_reuses_ocr_and_semantic(app, tmp_path):
    root, store, calls, run = app
    image(root / "one.png")
    run()
    calls.clear()
    other = tmp_path / "other"
    other.mkdir()
    (other / "copy.png").write_bytes((root / "one.png").read_bytes())
    result = run(root=str(other), collection="other", siglip_precision="fp32")
    assert result.reused_stages == {"metadata": 1, "ocr": 1, "vision": 0, "semantic": 1}
    assert counts(calls) == {"ocr": 0, "vision": 1, "semantic": 0}


def test_only_changed_stage_dependencies_invalidate_keys(app):
    root, store, calls, run = app
    image(root / "one.png")
    result = run()
    catalog = Catalog(store)
    spec = json.loads(catalog.job(result.job_id)["config"])["spec"]
    import copy

    for field, impacted in [
        ("languages", {"ocr", "semantic"}),
        ("ocr_sha256", {"ocr", "semantic"}),
        ("semantic_characters", {"semantic"}),
        ("thumbnail_side", {"metadata"}),
    ]:
        changed = copy.deepcopy(spec)
        changed[field] = "new" if isinstance(spec[field], str) else spec[field] + 1
        actual = {
            stage
            for stage in ("metadata", "ocr", "vision", "semantic")
            if stage_cache.dependencies(spec, stage) != stage_cache.dependencies(changed, stage)
        }
        assert actual == impacted
    for model, artifact, impacted in [
        ("siglip2", "vision_int8", {"vision"}),
        ("e5", "text_int8", {"semantic"}),
    ]:
        changed = copy.deepcopy(spec)
        changed["model_registry"]["models"][model]["files"][artifact]["sha256"] = "changed"
        assert {
            stage
            for stage in ("metadata", "ocr", "vision", "semantic")
            if stage_cache.dependencies(spec, stage) != stage_cache.dependencies(changed, stage)
        } == impacted
    catalog.close()


def test_compiled_module_uses_equivalent_shipped_source_identity(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 3\n", encoding="utf-8")
    compiled = tmp_path / "module.pyc"
    py_compile.compile(str(source), cfile=str(compiled), doraise=True)
    assert stage_cache.implementation(source) == stage_cache.implementation(compiled)
    before = stage_cache.implementation(compiled)
    source.write_text("VALUE = 4\n", encoding="utf-8")
    assert stage_cache.implementation(compiled) != before
    source.unlink()
    assert stage_cache.implementation(compiled) != before


def test_legacy_full_pipeline_rows_are_reused_when_dependencies_match(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    first = run()
    # Remove only the new aliases to simulate a version-2 catalog predating this feature.
    catalog = Catalog(store)
    config = json.loads(catalog.job(first.job_id)["config"])
    with catalog.db:
        catalog.db.execute("DELETE FROM analyses WHERE pipeline!=?", (config["pipeline"],))
    catalog.close()
    calls.clear()
    changed = run(languages="fra")
    assert changed.reused_stages["vision"] == 1 and changed.reused_stages["metadata"] == 1
    assert not [c for c in calls if c[0] == "vision"]


def test_semantic_cache_requires_matching_actual_ocr_text(app, monkeypatch):
    root, store, calls, run = app
    image(root / "one.png")
    first = run()
    catalog = Catalog(store)
    config = json.loads(catalog.job(first.job_id)["config"])
    key = catalog.files(first.job_id)[0]["image_id"]
    # Keep old semantic cache but replace OCR through the supported storage operation.
    catalog.save_analysis(
        config["pipeline"],
        key,
        "ocr",
        {"status": "ok", "text": "different text", "words": []},
        job_id=first.job_id,
    )
    catalog.close()
    calls.clear()
    result = run()
    assert result.processed_stages["semantic"] == 1
    assert [c for c in calls if c[0] == "semantic"] == [("semantic", "different text")]


def test_live_progress_separates_cache_hits_processed_failures_and_pending(app, monkeypatch):
    from realopen_lens.indexing import processing

    root, store, calls, run = app
    image(root / "one.png")
    run()
    image(root / "two.png", "green")
    observed = []
    original = processing.process_stage

    def observe(catalog, job_id, stage):
        before = service.collection_status("personal", store)["job"]
        assert before["phase"] == stage
        observed.append((stage, "before", before))
        original(catalog, job_id, stage)
        observed.append((stage, "after", service.collection_status("personal", store)["job"]))

    monkeypatch.setattr(processing, "process_stage", observe)
    run()
    for stage, when, progress in observed:
        assert progress["phase_cached_images"] == 1
        assert progress["phase_processed_images"] == (when == "after")
        assert progress["phase_completed_images"] == (2 if when == "after" else 1)
        assert progress["unique_images"] == 2


def test_failed_ocr_is_retried_without_invalidating_visual_work(app, monkeypatch):
    from realopen_lens.ocr import tesseract

    root, store, calls, run = app
    image(root / "one.png")
    original = tesseract.extract

    def fail(*args):
        raise TimeoutError("OCR timeout")

    monkeypatch.setattr(tesseract, "extract", fail)
    first = run()
    assert first.status == "completed_with_errors"
    calls.clear()
    monkeypatch.setattr(tesseract, "extract", original)
    result = run()
    assert result.status == "completed"
    assert result.reused_stages == {"metadata": 1, "ocr": 0, "vision": 1, "semantic": 0}
    assert counts(calls) == {"ocr": 1, "vision": 0, "semantic": 1}


def test_remembering_unchanged_success_does_not_rewrite_cache_rows(app):
    root, store, calls, run = app
    image(root / "one.png")
    result = run()
    catalog = Catalog(store)
    config = json.loads(catalog.job(result.job_id)["config"])
    before = catalog.db.total_changes
    for stage in ("metadata", "ocr", "vision", "semantic"):
        stage_cache.remember(catalog, result.job_id, config, stage)
    assert catalog.db.total_changes == before
    catalog.close()
