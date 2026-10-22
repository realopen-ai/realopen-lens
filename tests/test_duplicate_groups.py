"""Duplicate identity, review, scope and saved-reference regressions."""

from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageOps
import pytest

from test_collections import app as collection_fixture
from realopen_lens.models.manager import digest
from realopen_lens.organization import duplicates, collections
from realopen_lens.organization.fingerprints import fingerprint, near_distance
from realopen_lens.retrieval import search, images
from realopen_lens.learning import feedback

app = collection_fixture


def patterned(path: Path, size=(160, 120)):
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    with Image.fromarray(pixels).resize(size, Image.Resampling.BICUBIC) as image:
        image.save(path)


@pytest.fixture
def library(app, monkeypatch):
    root, store, calls, run = app
    patterned(root / "one.png")
    with Image.open(root / "one.png") as im:
        im.save(root / "same.bmp")
        im.resize((80, 60), Image.Resampling.LANCZOS).save(root / "small.png")
    Image.new("RGB", (160, 120), "green").save(root / "other.png")
    run()
    monkeypatch.setattr(search, "verify", lambda *a: None)
    monkeypatch.setattr(images, "verify_image_encoder", lambda *a: None)
    monkeypatch.setattr(
        search,
        "query_vector",
        lambda q, kind, *a: np.eye(1, 768 if kind == "siglip2" else 384, dtype=np.float32)[0],
    )
    return root, store, calls, run


def test_pixel_identity_defines_orientation_alpha_dimensions_and_frames(tmp_path):
    original = tmp_path / "original.png"
    patterned(original)
    with Image.open(original) as im:
        im.save(tmp_path / "same.bmp")
        flipped = ImageOps.mirror(im)
        flipped.save(tmp_path / "flip.png")
    assert fingerprint(original)["pixels"] == fingerprint(tmp_path / "same.bmp")["pixels"]
    assert fingerprint(original)["pixels"] != fingerprint(tmp_path / "flip.png")["pixels"]
    with Image.open(original) as source:
        rotated = source.transpose(Image.Transpose.ROTATE_90)
        exif = rotated.getexif()
        exif[274] = 6
        rotated.save(tmp_path / "oriented.png", exif=exif)
    assert fingerprint(original)["pixels"] == fingerprint(tmp_path / "oriented.png")["pixels"]
    a = Image.new("RGBA", (3, 2), (255, 0, 0, 0))
    b = Image.new("RGBA", (3, 2), (0, 255, 0, 0))
    a.save(tmp_path / "alpha-a.png")
    b.save(tmp_path / "alpha-b.png")
    assert (
        fingerprint(tmp_path / "alpha-a.png")["pixels"]
        == fingerprint(tmp_path / "alpha-b.png")["pixels"]
    )
    a.save(tmp_path / "animated.png", save_all=True, append_images=[b], duration=100)
    with pytest.raises(ValueError, match="Multi-frame"):
        fingerprint(tmp_path / "animated.png")


def test_scan_resumes_and_reviews_without_modifying_sources_or_model_stages(library):
    root, store, calls, run = library
    before = {p.name: digest(p) for p in root.iterdir()}
    original_calls = list(calls)
    events = []
    result = duplicates.scan(store, on_progress=events.append)
    assert result["status"] == "completed"
    assert result["fingerprinted"] == 4 and len(result["exact"]) == 1
    assert result["near"]
    assert events[0]["phase"] == "fingerprinting" and events[-1]["phase"] == "publishing"
    assert duplicates.scan(store)["reused"] == 4
    pair = result["near"][0]
    duplicates.review(store, "personal", pair["id"], "confirmed")
    page = duplicates.browse(store)
    assert any(g["state"] == "confirmed" for g in page["groups"])
    assert len(next(g for g in page["groups"] if g["kind"] == "exact")["results"]) == 2
    duplicates.scan(store)
    assert any(g["state"] == "confirmed" for g in duplicates.browse(store)["groups"])
    assert before == {p.name: digest(p) for p in root.iterdir()} and calls == original_calls


def test_renamed_folder_keeps_reviewed_near_duplicate_groups(library):
    from realopen_lens.organization.folders import change
    from realopen_lens.storage.catalog import Catalog

    root, store, calls, run = library
    scanned = duplicates.scan(store)
    pair = scanned["near"][0]["id"]
    duplicates.review(store, "personal", pair, "confirmed")
    catalog = Catalog(store, readonly=True)
    collection_id = catalog.collection("personal")["id"]
    catalog.close()
    change(store, "personal", collection_id, "Renamed")
    groups = duplicates.browse(store, "Renamed")["groups"]
    assert next(group for group in groups if group["id"] == pair)["state"] == "confirmed"
    duplicates.review(store, "Renamed", pair, "dismissed")
    assert (
        next(
            group for group in duplicates.browse(store, "Renamed")["groups"] if group["id"] == pair
        )["state"]
        == "dismissed"
    )


@pytest.mark.parametrize("mode", ["hybrid", "visual", "lexical", "semantic"])
def test_exact_results_backfill_and_preserve_feedback_and_scope(library, mode):
    root, store, calls, run = library
    duplicates.scan(store)
    results = search.search_collection("red", index_dir=store, mode=mode, limit=3)
    assert len(results) == (2 if mode == "lexical" else 3)
    merged = next(r for r in results if r.duplicate_ids)
    assert len(merged.paths) == 2 and len(merged.duplicate_ids) == 1
    group = collections.create(store, "personal", "Only alias")
    alias = merged.duplicate_ids[0]
    group = collections.change_members(store, "personal", group["id"], [alias], 1)
    scoped = search.search_collection(
        "red", index_dir=store, mode=mode, virtual_collection=group["id"]
    )
    assert [r.image_id for r in scoped] == [alias]
    assert not scoped[0].duplicate_ids and len(scoped[0].paths) == 1
    saved = feedback.search_and_record("red", index_dir=store, mode=mode, limit=3)
    feedback.judge(saved.run_id, merged.image_id, "relevant", index_dir=store)
    stored = feedback.saved_search(saved.run_id, index_dir=store)
    assert next(r for r in stored["results"] if r["image_id"] == merged.image_id)["duplicate_ids"]
    assert collections.browse(store, "personal", group["id"])["member_count"] == 1


def test_self_exclusion_removes_pixel_aliases_but_not_near_matches(library):
    root, store, calls, run = library
    duplicates.scan(store)
    result = images.search_similar(image_id=digest(root / "one.png"), index_dir=store)
    ids = {r.image_id for r in result}
    assert digest(root / "same.bmp") not in ids
    assert digest(root / "small.png") in ids
    assert (
        len(
            images.search_similar(
                image_id=digest(root / "one.png"), index_dir=store, include_self=True
            )
        )
        == 3
    )


def test_interruption_reuses_completed_fingerprints_without_publishing(library):
    root, store, calls, run = library

    def stop(event):
        if event["phase"] == "fingerprinting" and event["step"] == 2:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        duplicates.scan(store, on_progress=stop)
    assert duplicates.browse(store)["scanned"] is False
    assert duplicates.scan(store)["reused"] == 2


def test_new_generation_requires_rescan_and_unavailable_aliases_are_not_returned(library):
    root, store, calls, run = library
    duplicates.scan(store)
    (root / "same.bmp").unlink()
    result = search.search_collection("red", index_dir=store, mode="lexical")
    assert all(not r.duplicate_ids for r in result)
    run()
    assert duplicates.browse(store)["scanned"] is False
    assert duplicates.scan(store)["reused"] == 3


def test_low_information_and_changed_diagrams_are_not_near_duplicates(tmp_path):
    for name, color in [("red", "red"), ("green", "green")]:
        Image.new("RGB", (160, 120), color).save(tmp_path / (name + ".png"))
    assert (
        near_distance(fingerprint(tmp_path / "red.png"), fingerprint(tmp_path / "green.png"))
        is None
    )


def test_cli_duplicate_lifecycle(library):
    from typer.testing import CliRunner
    from realopen_lens.cli import app

    root, store, calls, run = library
    runner = CliRunner()
    scanned = runner.invoke(app, ["duplicates", "scan", "--index-dir", str(store)])
    assert scanned.exit_code == 0, scanned.output
    pair = json.loads(scanned.stdout)["near"][0]
    result = runner.invoke(
        app, ["duplicates", "review", pair["id"], "dismissed", "--index-dir", str(store)]
    )
    assert result.exit_code == 0 and json.loads(result.stdout)["state"] == "dismissed"


def test_corrupt_derived_fingerprint_is_rebuilt_and_export_cannot_overwrite_it(library):
    root, store, calls, run = library
    duplicates.scan(store)
    path = next((store / "duplicates/cache").glob("*.json"))
    path.write_text('{"bad":', encoding="utf-8")
    scanned = duplicates.scan(store)
    assert scanned["reused"] == 3 and not scanned["errors"]
    with pytest.raises(ValueError, match="duplicate fingerprints"):
        feedback.export_feedback(path, index_dir=store, overwrite=True)


def test_similarity_scope_does_not_allow_query_alias_back_in(library):
    root, store, calls, run = library
    duplicates.scan(store)
    group = collections.create(store, "personal", "Alias")
    collections.change_members(store, "personal", group["id"], [digest(root / "same.bmp")], 1)
    assert (
        images.search_similar(
            image_id=digest(root / "one.png"), index_dir=store, virtual_collection=group["id"]
        )
        == []
    )


def test_external_reencoded_query_excludes_same_pixels(library, tmp_path, monkeypatch):
    from PIL.PngImagePlugin import PngInfo

    root, store, calls, run = library
    duplicates.scan(store)
    query = tmp_path / "external.png"
    info = PngInfo()
    info.add_text("comment", "new byte identity, same pixels")
    with Image.open(root / "one.png") as source:
        source.save(query, pnginfo=info)
    assert digest(query) not in {digest(p) for p in root.iterdir()}
    monkeypatch.setattr(images, "image_vector", lambda *a: np.eye(1, 768, dtype=np.float32)[0])
    result = images.search_similar(query, index_dir=store)
    assert digest(root / "one.png") not in {r.image_id for r in result}
    assert digest(root / "same.bmp") not in {r.image_id for r in result}
    assert digest(root / "small.png") in {r.image_id for r in result}


def test_collapsed_search_only_reads_thumbnails_for_displayed_results(library, monkeypatch):
    root, store, calls, run = library
    duplicates.scan(store)
    read = []
    original = search.digest

    def counted(path):
        if path.name == "thumbnail.jpg":
            read.append(path)
        return original(path)

    monkeypatch.setattr(search, "digest", counted)
    assert len(search.search_collection("red", index_dir=store, mode="visual", limit=1)) == 1
    assert len(read) == 1
    read.clear()
    assert (
        len(images.search_similar(image_id=digest(root / "one.png"), index_dir=store, limit=1)) == 1
    )
    assert len(read) == 1
