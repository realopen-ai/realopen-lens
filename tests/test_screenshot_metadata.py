"""Screenshot evidence is positive metadata, never an inferred negative label."""

from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from test_collections import app as _app, image
from realopen_lens.metadata.exif import extract, user_comment
from realopen_lens.retrieval import metadata, search
from realopen_lens.storage.catalog import Catalog

app = _app


def screenshot(path: Path, comment=b"ASCII\0\0\0Screenshot", description=None):
    tags = Image.Exif()
    tags[34665] = {37510: comment}
    if description:
        tags[270] = description
    Image.new("RGB", (64, 48), "#dd9955").save(path, exif=tags)


@pytest.mark.parametrize(
    "comment",
    [
        b"ASCII\0\0\0Screenshot\0",
        b"UNICODE\0" + "Screenshot".encode("utf-16"),
        b"UNICODE\0" + "Screenshot".encode("utf-16-be"),
        b"UNICODE\0" + "Screenshot".encode("utf-16-le"),
        "Screenshot",
    ],
)
def test_user_comment_charsets_are_decoded_without_guessing_from_dimensions(comment):
    assert user_comment(comment) == "Screenshot"
    assert user_comment(b"UNICODE\0odd") is None


def test_jpeg_markers_preserve_sources_and_plain_images_remain_unknown(tmp_path):
    path = tmp_path / "export.jpg"
    screenshot(path, description="Screenshot")
    with Image.open(path) as source:
        value = extract(source)
    assert value["screenshot"]["state"] == "marked"
    assert {item["source"] for item in value["screenshot"]["evidence"]} == {
        "EXIF.UserComment",
        "EXIF.ImageDescription",
    }
    assert value["user_comment"] == "Screenshot"
    screenshot(
        path, comment=b"ASCII\0\0\0Not a screenshot", description="An article about screenshots"
    )
    with Image.open(path) as source:
        assert extract(source)["screenshot"] == {"state": "unknown", "evidence": []}
    path = tmp_path / "Screenshot 2026-10-25.png"
    Image.new("RGB", (1170, 2532), "white").save(path)
    with Image.open(path) as source:
        assert extract(source)["screenshot"]["state"] == "unknown"


def test_png_text_and_xmp_markers_are_namespace_specific_and_bounded(tmp_path):
    xml = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">Screenshot</rdf:li>'
        "</rdf:Alt></dc:description></x:xmpmeta>"
    )
    path = tmp_path / "capture.png"
    info = PngImagePlugin.PngInfo()
    info.add_itxt("XML:com.adobe.xmp", xml)
    info.add_text("Description", "Screenshot")
    Image.new("RGB", (80, 60)).save(path, pnginfo=info)
    with Image.open(path) as source:
        assert {item["source"] for item in extract(source)["screenshot"]["evidence"]} == {
            "PNG.Description",
            "XMP.dc.description",
        }
    for xml in (
        xml.replace("http://purl.org/dc/elements/1.1/", "https://unrelated.example/"),
        '<!DOCTYPE a [<!ENTITY capture "Screenshot">]><a>&capture;</a>',
        "x" * 65537,
        "<broken>",
    ):
        info = PngImagePlugin.PngInfo()
        info.add_itxt("XML:com.adobe.xmp", xml)
        Image.new("RGB", (80, 60)).save(path, pnginfo=info)
        with Image.open(path) as source:
            assert extract(source)["screenshot"]["state"] == "unknown"


def rows():
    return [
        {"metadata": {"exif": {"schema": 1, "screenshot": {"state": state, "evidence": []}}}}
        for state in ("marked", "unknown")
    ]


def test_normal_screenshot_queries_preserve_unknowns_and_scene_ranking():
    plan = metadata.compile_query("screenshots from last week", rows(), today=date(2026, 10, 25))
    screenshot_rule = next(rule for rule in plan.constraints if rule["kind"] == "screenshot")
    assert screenshot_rule["policy"] == "markers_and_visual" and screenshot_rule["promote"]
    assert plan.text == "screenshots"
    lookup = {"marked": rows()[0], "unknown": rows()[1]}
    assert metadata.screenshot_ranking(plan, ["unknown"], lookup, {}) == ["marked", "unknown"]
    scene = metadata.compile_query("screenshots of invoices", rows())
    assert metadata.screenshot_ranking(scene, ["unknown", "marked"], lookup, {}) == [
        "unknown",
        "marked",
    ]
    absent = metadata.compile_query("screenshots", [rows()[1]])
    assert not absent.constraints and absent.text == "screenshots" and absent.notices
    assert not metadata.compile_query("without screenshots", rows()).constraints


def test_marked_only_query_filters_before_limits_without_models(app, monkeypatch):
    root, store, calls, run = app
    screenshot(root / "tagged.jpg")
    for index in range(55):
        image(root / f"normal-{index}.png", (index, 20, 30))
    run()
    monkeypatch.setattr(
        search, "verify", lambda *a: pytest.fail("Metadata-only query loaded models")
    )
    monkeypatch.setattr(
        search, "query_vector", lambda *a: pytest.fail("Metadata-only query encoded")
    )
    results = search.search_collection("marked screenshots", index_dir=store, limit=1)
    assert len(results) == 1 and results[0].paths[0].endswith("tagged.jpg")


def test_broad_screenshot_request_includes_markers_outside_channel_candidates(app, monkeypatch):
    root, store, calls, run = app
    screenshot(root / "tagged.jpg")
    image(root / "unknown.png")
    run()
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(
        search,
        "query_vector",
        lambda text, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0],
    )
    catalog = Catalog(store, readonly=True)
    try:
        from realopen_lens.api.collections import browse

        results = browse(store, "personal", 0, 10)["results"]
        unknown = next(
            row["image_id"] for row in results if row["paths"][0].endswith("unknown.png")
        )
    finally:
        catalog.close()
    monkeypatch.setattr(search, "hybrid_rank", lambda *a, **kw: [unknown])
    result = search.search_collection("screenshots", index_dir=store, limit=2, use_feedback=False)
    assert result[0].paths[0].endswith("tagged.jpg")
    assert result[1].paths[0].endswith("unknown.png")


def test_marker_preference_keeps_existing_exact_query_feedback(app, monkeypatch):
    from realopen_lens.learning import feedback
    from realopen_lens.storage.feedback import memory_scope

    root, store, calls, run = app
    screenshot(root / "tagged.jpg")
    image(root / "unknown.png")
    run()
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(
        search,
        "query_vector",
        lambda text, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0],
    )
    recorded = feedback.search_and_record("screenshots", index_dir=store, interpret_metadata=False)
    unknown = next(row for row in recorded.results if row.paths[0].endswith("unknown.png"))
    feedback.change_judgment(recorded.run_id, unknown.image_id, "relevant", index_dir=store)
    plan = metadata.compile_query("screenshots", rows())
    assert memory_scope("", plan.report()) == ""
    result = search.search_collection("screenshots", index_dir=store)
    assert result[0].image_id == unknown.image_id
    assert result[0].feedback["label"] == "relevant"


def test_library_marker_preference_strict_mode_and_scope(app, monkeypatch):
    from realopen_lens.retrieval.library import LibrarySession
    from realopen_lens.runtime.resident import ResidentQueryEncoder
    from realopen_lens.organization import collections

    root, store, calls, run = app
    screenshot(root / "tagged.jpg")
    image(root / "unknown.png")
    run()
    monkeypatch.setattr(ResidentQueryEncoder, "verify", lambda *a: None)
    monkeypatch.setattr(
        ResidentQueryEncoder,
        "encode",
        lambda self, text, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[
            0
        ],
    )
    session = LibrarySession(index_dir=store)
    try:
        result = session.search("screenshots", use_feedback=False)
        assert len(result["results"]) == 2 and result["results"][0]["paths"][0].endswith(
            "tagged.jpg"
        )
        strict = session.search("marked screenshots")
        assert len(strict["results"]) == 1 and strict["ranking"]["kind"] == "metadata"
        unknown = result["results"][1]["image_id"]
        group = collections.create(store, "personal", "Unknown only", all_folders=True)
        group = collections.change_members(
            store, "personal", group["id"], [unknown], group["revision"], all_folders=True
        )
        scoped = session.search(
            "marked screenshots", virtual_collection=group["id"], virtual_revision=group["revision"]
        )
        assert scoped["results"] == []
    finally:
        session.close()


def test_compact_xmp_attribute_is_an_explicit_marker(tmp_path):
    xml = '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    xml += 'xmlns:dc="http://purl.org/dc/elements/1.1/">'
    xml += '<rdf:Description dc:description="Screenshot"/></rdf:RDF>'
    info = PngImagePlugin.PngInfo()
    info.add_itxt("XML:com.adobe.xmp", xml)
    path = tmp_path / "compact.png"
    Image.new("RGB", (80, 60)).save(path, pnginfo=info)
    with Image.open(path) as source:
        assert extract(source)["screenshot"]["evidence"] == [
            {"source": "XMP.dc.description", "value": "Screenshot"}
        ]


@pytest.mark.parametrize("suffix", ["jpg", "heic"])
def test_committed_screenshot_markers_survive_heic_conversion(suffix):
    path = Path(__file__).parent / "fixtures/metadata" / f"screenshot-marker.{suffix}"
    with Image.open(path) as source:
        value = extract(source)
    assert value["screenshot"]["state"] == "marked"
    sources = {item["source"] for item in value["screenshot"]["evidence"]}
    assert ("EXIF.UserComment" if suffix == "jpg" else "XMP.dc.description") in sources
    assert value["captured_at"] == "2026-10-14T12:00:00+00:00"


def test_heic_conversion_with_stripped_marker_stays_unknown():
    path = Path(__file__).parent / "fixtures/metadata/screenshot-stripped.heic"
    with Image.open(path) as image:
        value = extract(image)
    assert value["captured_at"] == "2026-10-14T12:00:00+00:00"
    assert value["screenshot"] == {"state": "unknown", "evidence": []}
