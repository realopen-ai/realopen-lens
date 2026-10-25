"""Real EXIF/HEIC fixtures, candidate filtering and natural-language interpretation."""

from datetime import date
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from test_collections import app as _app, image
from realopen_lens.indexing import processing, service, stage_cache
from realopen_lens.metadata.exif import extract, number, coordinate, timestamp
from realopen_lens.metadata import places
from realopen_lens.retrieval import metadata, search
from realopen_lens.storage.catalog import Catalog

app = _app
FIXTURES = Path(__file__).parent / "fixtures/metadata"


def row(camera="iPhone 11 Pro Max", lat=48.8566, lon=2.3522, captured="2024-06-15T19:30:00+02:00"):
    return {
        "metadata": {
            "width": 3000,
            "height": 2000,
            "bytes": 4_000_000,
            "format": "JPEG",
            "exif": {
                "schema": 1,
                "camera_make": "Apple",
                "camera_model": camera,
                "captured_at": captured,
                "gps": {"latitude": lat, "longitude": lon},
            },
        }
    }


@pytest.mark.parametrize("name", ["camera-gps.jpg", "camera-gps.heic"])
def test_real_jpeg_and_heic_keep_camera_date_gps_and_decode(name):
    data, thumbnail = processing.metadata(FIXTURES / name)
    assert data["width"] == 96 and data["height"] == 64
    assert data["exif"]["camera_model"] == "iPhone 11 Pro Max"
    assert data["exif"]["captured_at"] == "2024-06-15T19:30:00+02:00"
    assert data["exif"]["capture_timezone_known"]
    assert data["exif"]["gps"]["latitude"] == pytest.approx(48.85666667)
    assert data["exif"]["gps"]["longitude"] == pytest.approx(2.35222222)
    assert data["exif"]["iso"] == 100 and thumbnail.startswith(b"\xff\xd8")
    json.dumps(data, allow_nan=False)


def test_invalid_and_absent_exif_values_stay_unknown():
    with Image.new("RGB", (12, 12)) as im:
        value = extract(im)
        assert value["camera_model"] is None and value["gps"] is None
    assert number(float("nan")) is None and number((1, 0)) is None
    assert coordinate((90, 1, 0), "N", 90) is None
    assert coordinate((12, 60, 0), "N", 90) is None
    assert coordinate((12, 1, 0), "E", 90) is None
    assert coordinate((12, 1, 0), "S", 90) < 0
    assert timestamp("2024:06:15 12:00:00") == "2024-06-15T12:00:00"
    assert timestamp("2024:02:30 12:00:00") is None
    assert timestamp("2024:06:15 12:00:00", "+99:00") == "2024-06-15T12:00:00"


@pytest.mark.parametrize(
    "query,remaining,kinds",
    [
        ("sunset taken by iPhone 11 Pro Max", "sunset", {"camera"}),
        ("sunset taken with my iPhone 11 Pro Max in June 2024", "sunset", {"camera", "date"}),
        ("photos in Paris, France in June 2024", "", {"location", "date"}),
        ("sunset near 48.8566, 2.3522", "sunset", {"location"}),
        (
            'PNG diagrams from folder "Work" at least 2000 pixels wide',
            "diagrams",
            {"folder", "format", "dimension"},
        ),
        ("photos prises avec mon iPhone 11 Pro Max en juin 2024", "", {"camera", "date"}),
        ("صور بواسطة iPhone 11 Pro Max في يونيو 2024", "", {"camera", "date"}),
    ],
)
def test_natural_language_query_keeps_visual_content(query, remaining, kinds):
    plan = metadata.compile_query(query, [row()])
    assert plan.text == remaining
    assert {rule["kind"] for rule in plan.constraints} == kinds
    assert plan.report()["original"] == query
    if {"camera", "date"} <= kinds:
        assert (
            next(rule for rule in plan.constraints if rule["kind"] == "date")["basis"] == "capture"
        )


def test_unconstrained_descriptions_are_unchanged():
    for query in (
        "pizza",
        "a cat in a room",
        "woman with a camera",
        "ERR_42",
        "form 2024",
        "sunset landscape",
    ):
        plan = metadata.compile_query(query, [row()])
        assert plan.text == query and not plan.constraints


def test_places_are_offline_multilingual_and_disambiguated_with_available_gps():
    assert places.resolve("Paris, France", [], 25)["country"] == "FR"
    assert places.resolve("باريس", [row()["metadata"]["exif"]["gps"]], 25)["country"] == "FR"
    assert places.resolve("unknown-place-abcdef", [], 25) is None
    with pytest.raises(ValueError, match="Ambiguous place"):
        places.resolve("Paris", [], 25)
    assert places.distance(0, 179.99, 0, -179.99) < 3


def test_metadata_rules_are_conjunctive_and_unknown_values_do_not_match():
    plan = metadata.compile_query(
        "sunset taken with iPhone 11 Pro Max in June 2024 in Paris, France", [row()]
    )
    assert plan.matches(row()["metadata"], [])
    assert not plan.matches(row(camera="iPhone 12")["metadata"], [])
    assert not plan.matches(row(captured="2023-06-15T00:00:00")["metadata"], [])
    assert not plan.matches(row(lat=35, lon=139)["metadata"], [])
    assert not plan.matches({"exif": {}}, [])


def test_relative_dates_are_calendar_ranges_and_capture_dates_do_not_guess_timezone():
    plan = metadata.compile_query("screenshots from last week", [row()], today=date(2026, 10, 25))
    rule = plan.constraints[0]
    assert rule["start"] == "2026-10-12" and rule["end"] == "2026-10-19"
    assert plan.text == "screenshots"
    from datetime import datetime

    assert plan.matches({"exif": {}}, [datetime(2026, 10, 14)])
    strict = metadata.compile_query("photos taken in June 2024", [row()])
    assert not strict.matches({"exif": {}}, [datetime(2024, 6, 14)])


def test_collection_filters_before_candidate_selection_and_strips_camera_from_encoder(
    app, monkeypatch
):
    root, store, calls, run = app
    shutil.copyfile(FIXTURES / "camera-gps.jpg", root / "sunset.jpg")
    for index in range(55):
        image(root / f"other-{index}.png", (index, 20, 30))
    run()
    encoded = []

    def vector(text, kind, *args):
        encoded.append(text)
        return np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0]

    monkeypatch.setattr(search, "query_vector", vector)
    monkeypatch.setattr(search, "verify", lambda *args: None)
    trace, interpreted = {}, {}
    results = search.search_collection(
        "sunset taken with my iPhone 11 Pro Max",
        index_dir=store,
        trace=trace,
        metadata_trace=interpreted,
        use_feedback=False,
    )
    assert len(results) == 1 and results[0].paths[0].endswith("sunset.jpg")
    assert encoded == ["sunset", "sunset"]
    assert trace["searchable_images"] == 1 and interpreted["visual_query"] == "sunset"


def test_metadata_only_and_no_match_queries_do_not_load_models(app, monkeypatch):
    root, store, calls, run = app
    shutil.copyfile(FIXTURES / "camera-gps.heic", root / "phone.heic")
    image(root / "other.png")
    run()
    monkeypatch.setattr(
        search, "query_vector", lambda *a: pytest.fail("Metadata-only query encoded")
    )
    monkeypatch.setattr(
        search, "verify", lambda *a: pytest.fail("Metadata-only query verified models")
    )
    assert (
        len(search.search_collection("photos taken with iPhone 11 Pro Max", index_dir=store)) == 1
    )
    assert search.search_collection("photos taken with Imaginary Camera", index_dir=store) == []
    assert len(search.search_collection("PNG photos", index_dir=store)) == 1


def test_legacy_metadata_backfill_reuses_ocr_vision_and_semantic(app, monkeypatch):
    root, store, calls, run = app
    shutil.copyfile(FIXTURES / "camera-gps.jpg", root / "one.jpg")
    original = service.pipeline_config

    def old_config(options):
        config = original(options)
        spec = config["spec"]
        for key in ("processing_identity", "metadata_sha256", "heif_decoder"):
            spec.pop(key)
        spec["processing_sha256"] = next(iter(stage_cache._LEGACY_PROCESSING))
        config["pipeline"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        return config

    original_metadata = processing.metadata

    def old_metadata(path):
        result, thumb = original_metadata(path)
        result.pop("exif")
        return result, thumb

    with monkeypatch.context() as context:
        context.setattr(service, "pipeline_config", old_config)
        context.setattr(processing, "metadata", old_metadata)
        run()
    calls.clear()
    result = run()
    assert result.processed_stages == {"metadata": 1, "ocr": 0, "vision": 0, "semantic": 0}
    assert not calls
    catalog = Catalog(store, readonly=True)
    try:
        from realopen_lens.storage.artifacts import load_manifest

        manifest = load_manifest(catalog, catalog.snapshot("personal")[1])
        assert manifest["images"][0]["metadata"]["exif"]["camera_model"] == "iPhone 11 Pro Max"
    finally:
        catalog.close()


def test_city_names_printed_in_documents_are_content_not_capture_locations():
    for query in (
        "Publicité de l’Hotel Kalee à Shanghai mentionnant Cuisine Française.",
        "Advertisement for a hotel in San Francisco",
        "إعلان عن فندق في سان فرانسيسكو",
        "Diagramme avec Statement API, Sign In API et Database.",
        "Photo d’hommes à une table de pub avec des verres.",
    ):
        plan = metadata.compile_query(query, [row()])
        assert not plan.constraints and plan.text == query
    assert metadata.compile_query("document photographed in Paris, France", [row()]).constraints


def test_named_city_radius_and_implicit_known_camera_preserve_scene_words():
    plan = metadata.compile_query(
        "sunset with iPhone 11 Pro Max within 5 km of Paris, France", [row()]
    )
    assert plan.text == "sunset"
    assert {rule["kind"] for rule in plan.constraints} == {"camera", "location"}
    assert plan.constraints[-1]["radius_km"] == 5
    assert (
        metadata.compile_query("sunset taken with iPhone 11 Pro Max at the beach", [row()]).text
        == "sunset at the beach"
    )


def test_filtered_feedback_has_its_own_namespace_and_hidden_browsing(app, monkeypatch):
    from realopen_lens.learning import feedback
    from realopen_lens.retrieval import memory
    from realopen_lens.storage.feedback import memory_key

    root, store, calls, run = app
    shutil.copyfile(FIXTURES / "camera-gps.jpg", root / "phone.jpg")
    image(root / "other.png")
    run()
    query = "photos taken with iPhone 11 Pro Max"
    recorded = feedback.search_and_record(query, index_dir=store)
    assert len(recorded.results) == 1
    saved = feedback.saved_search(recorded.run_id, index_dir=store)
    assert saved["ranking"]["query_interpretation"]["constraints"][0]["kind"] == "camera"
    assert ":metadata-v1:" in memory_key(saved)[2]
    feedback.change_judgment(
        recorded.run_id, recorded.results[0].image_id, "relevant", index_dir=store
    )
    from realopen_lens.learning.review import audit

    report = audit(store)
    assert any(
        "metadata" in reason.lower() for entry in report["runs"] for reason in entry["reasons"]
    )
    feedback.change_judgment(
        recorded.run_id, recorded.results[0].image_id, "irrelevant", index_dir=store
    )
    assert search.search_collection(query, index_dir=store) == []
    assert len(memory.hidden(store, query)["results"]) == 1
    assert memory.hidden(store, query, interpret_metadata=False)["total"] == 0


def test_library_metadata_queries_filter_before_fusion_and_keep_scope(app, tmp_path):
    from realopen_lens.retrieval.library import LibrarySession
    from realopen_lens.organization import collections

    root, store, calls, run = app
    shutil.copyfile(FIXTURES / "camera-gps.jpg", root / "phone.jpg")
    run()
    other = tmp_path / "other"
    other.mkdir()
    image(other / "other.png")
    run(root=str(other), collection="other")
    library = LibrarySession(index_dir=store)
    try:
        result = library.search("photos taken with iPhone 11 Pro Max")
        assert len(result["results"]) == 1
        assert result["interpretation"]["visual_query"] == ""
        assert result["ranking"]["kind"] == "metadata"
        assert result["results"][0]["sources"][0]["collection"] == "personal"
        scoped = collections.create(store, "personal", "Other only", all_folders=True)
        catalog = Catalog(store, readonly=True)
        try:
            from realopen_lens.api.collections import browse

            key = browse(store, "other", 0, 10)["results"][0]["image_id"]
        finally:
            catalog.close()
        scoped = collections.change_members(
            store, "personal", scoped["id"], [key], scoped["revision"], all_folders=True
        )
        result = library.search(
            "photos taken with iPhone 11 Pro Max",
            virtual_collection=scoped["id"],
            virtual_revision=scoped["revision"],
        )
        assert result["results"] == []
    finally:
        library.close()


def test_decoder_change_cannot_reuse_old_heif_stages_through_full_pipeline_fallback(
    app, monkeypatch
):
    root, store, calls, run = app
    shutil.copyfile(FIXTURES / "camera-gps.heic", root / "phone.heic")
    image(root / "normal.png")
    run()
    calls.clear()
    original_version = service.version
    monkeypatch.setattr(
        service,
        "version",
        lambda name: "changed-decoder" if name == "pi-heif" else original_version(name),
    )
    result = run()
    assert result.processed_stages == {"metadata": 2, "ocr": 1, "vision": 1, "semantic": 1}
    assert result.reused_stages == {"metadata": 0, "ocr": 1, "vision": 1, "semantic": 1}
