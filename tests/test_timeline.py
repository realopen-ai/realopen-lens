"""Chronology, scope isolation and stable model-free browsing across page boundaries."""

from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import pytest

from test_collections import app as _app, image
from realopen_lens.api import collections as reading
from realopen_lens.metadata import timeline
from realopen_lens.organization import collections as virtual
from realopen_lens.retrieval.library import LibrarySession
from realopen_lens.storage.organization import CollectionConflict

app = _app


def test_capture_calendar_dates_keep_offsets_and_unknown_timezones(tmp_path):
    for captured, known in [("2024-01-01T00:15:00+14:00", True), ("2024-01-01T00:15:00", False)]:
        row = {"metadata": {"exif": {"captured_at": captured}}}
        value = timeline.image_date(row, [], tmp_path)
        assert value["day"] == "2024-01-01" and value["month"] == "2024-01"
        assert value["source"] == "capture" and value["timezone_known"] is known
        assert value["date"] == captured


def test_fallback_uses_only_available_indexed_signatures_and_reports_unknown(tmp_path):
    stamp = int(datetime(2023, 12, 31, 23, 30, tzinfo=timezone.utc).timestamp() * 1e9)
    row = {
        "metadata": {"exif": {"captured_at": "broken"}},
        "signatures": {"a.png": {"mtime_ns": stamp}, "b.png": {"mtime_ns": stamp * 2}},
    }
    value = timeline.image_date(row, [str(tmp_path / "a.png")], tmp_path)
    assert value["source"] == "modified" and value["day"] == "2023-12-31"
    assert timeline.image_date(row, [], tmp_path)["source"] == "unknown"
    row["signatures"]["a.png"]["mtime_ns"] = 10**100
    assert timeline.image_date(row, [str(tmp_path / "a.png")], tmp_path)["date"] is None


def test_order_is_stable_and_unknown_dates_stay_last_in_both_directions():
    dates = {
        "c": {"date": None},
        "b": {"date": "2024-01-01T00:00:00"},
        "a": {"date": "2024-01-01T00:00:00"},
        "d": {"date": "2025-01-01T00:00:00"},
    }
    assert timeline.ordered(dates, "newest") == ["d", "a", "b", "c"]
    assert timeline.ordered(dates, "oldest") == ["a", "b", "d", "c"]
    with pytest.raises(ValueError):
        timeline.ordered(dates, "random")


def populate(app):
    root, store, calls, run = app
    fixture = Path(__file__).parent / "fixtures/metadata/camera-gps.jpg"
    shutil.copyfile(fixture, root / "capture.jpg")
    image(root / "old.png", "red")
    image(root / "new.png", "green")
    for name, year in [("old.png", 2020), ("new.png", 2025), ("capture.jpg", 2026)]:
        stamp = datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()
        os.utime(root / name, (stamp, stamp))
    run()
    return root, store, calls, run


def test_folder_pages_cover_every_image_once_and_retain_capture_priority(app):
    root, store, calls, run = populate(app)
    first = reading.browse(store, "personal", 0, 2)
    second = reading.browse(
        store,
        "personal",
        first["next_offset"],
        2,
        generation=first["generation"],
        expected_snapshot=first["snapshot"],
    )
    records = first["results"] + second["results"]
    assert [Path(row["paths"][0]).name for row in records] == ["new.png", "capture.jpg", "old.png"]
    assert [row["rank"] for row in records] == [1, 2, 3]
    assert [row["timeline"]["source"] for row in records] == ["modified", "capture", "modified"]
    assert second["next_offset"] is None and second["snapshot"] == first["snapshot"]
    assert reading.browse(store, "personal", 0, 3, order="oldest")["results"] == [
        {**row, "rank": i + 1} for i, row in enumerate(reversed(records))
    ]


@pytest.mark.parametrize("change", ["generation", "availability", "order"])
def test_folder_page_rejects_changed_snapshot(app, change):
    root, store, calls, run = populate(app)
    first = reading.browse(store, "personal", 0, 1)
    if change == "generation":
        image(root / "added.png", "blue")
        run()
    elif change == "availability":
        (root / "old.png").unlink()
    with pytest.raises(CollectionConflict):
        reading.browse(
            store,
            "personal",
            1,
            1,
            generation=first["generation"],
            expected_snapshot=first["snapshot"],
            order="oldest" if change == "order" else "newest",
        )


def test_virtual_timeline_never_includes_nonmembers_and_detects_revision(app):
    root, store, calls, run = populate(app)
    rows = reading.browse(store, "personal", 0, 10)["results"]
    group = virtual.create(store, "personal", "Two")
    group = virtual.change_members(
        store,
        "personal",
        group["id"],
        [rows[0]["image_id"], rows[2]["image_id"]],
        group["revision"],
    )
    first = virtual.browse(store, "personal", group["id"], limit=1)
    last = virtual.browse(
        store,
        "personal",
        group["id"],
        offset=1,
        limit=1,
        generation=first["generation"],
        expected_snapshot=first["snapshot"],
    )
    assert [first["results"][0]["image_id"], last["results"][0]["image_id"]] == [
        rows[0]["image_id"],
        rows[2]["image_id"],
    ]
    assert last["next_offset"] is None
    virtual.change_members(store, "personal", group["id"], [rows[1]["image_id"]], group["revision"])
    with pytest.raises(CollectionConflict):
        virtual.browse(
            store, "personal", group["id"], offset=1, expected_snapshot=first["snapshot"]
        )


def test_library_timeline_deduplicates_copies_without_models_or_lexical_index(
    app, tmp_path, monkeypatch
):
    root, store, calls, run = populate(app)
    other = tmp_path / "other"
    other.mkdir()
    shutil.copyfile(root / "capture.jpg", other / "copy.jpg")
    image(other / "blue.png", "blue")
    stamp = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(other / "blue.png", (stamp, stamp))
    run(root=str(other), collection="other", siglip_precision="fp32")

    def forbidden(*args, **kwargs):
        raise AssertionError("Browsing must not load models or build a text index")

    monkeypatch.setattr("realopen_lens.retrieval.library.ResidentQueryEncoder", forbidden)
    monkeypatch.setattr("realopen_lens.retrieval.ranking.lexical_index", forbidden)
    with closing(LibrarySession(index_dir=store)) as session:
        first = session.browse(limit=2)
        last = session.browse(
            offset=2, limit=2, generation=first["generation"], expected_snapshot=first["snapshot"]
        )
        records = first["results"] + last["results"]
        assert first["total"] == 4 and len({r["image_id"] for r in records}) == 4
        assert [r["timeline"]["day"][:4] for r in records] == ["2025", "2024", "2022", "2020"]
        assert len(records[1]["sources"]) == 2 and records[1]["timeline"]["source"] == "capture"
        assert last["next_offset"] is None and last["results"][0]["rank"] == 3
        with pytest.raises(CollectionConflict):
            session.browse(offset=2, browse_order="oldest", expected_snapshot=first["snapshot"])
        (other / "blue.png").unlink()
        with pytest.raises(CollectionConflict):
            session.browse(offset=2, generation=first["generation"])
