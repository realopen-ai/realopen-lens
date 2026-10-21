"""Workspace import preserves history, rejects conflicts and publishes only verified copies."""

from pathlib import Path
import json
import shutil
import sqlite3

import pytest
from typer.testing import CliRunner

from test_virtual_collections import library as library_fixture
from test_collections import app as collection_fixture
from realopen_lens.cli import app as cli
from realopen_lens.learning import feedback
from realopen_lens.organization import collections, duplicates
from realopen_lens.storage.feedback import FeedbackStore
from realopen_lens.workspace.service import preview, initialize
from realopen_lens.workspace.identity import describe, models_directory
from realopen_lens.retrieval import search

app = collection_fixture
library = library_fixture


def snapshot(root):
    result = {}
    for name in ("catalog.sqlite3", "feedback.sqlite3", "organization.sqlite3"):
        if (root / name).is_file():
            with sqlite3.connect((root / name).resolve().as_uri() + "?mode=ro", uri=True) as db:
                result[name] = list(db.iterdump())
    return result


def imported(sources, destination, **kwargs):
    plan = preview(sources, destination)
    return initialize(sources, destination, plan["plan_sha256"], models=Path("models"), **kwargs)


def test_import_combines_compatible_histories_and_virtuals_without_touching_originals(
    library, tmp_path
):
    root, source, _, _ = library
    record = feedback.search_and_record("red", index_dir=source, use_feedback=False)
    key = record.results[0].image_id
    feedback.judge(record.run_id, key, "relevant", index_dir=source)
    group = collections.create(source, "personal", "Reviewed images")
    collections.change_members(source, "personal", group["id"], [key], group["revision"])
    other = tmp_path / "other-index"
    shutil.copytree(source, other)
    # A shared history prefix followed by an explicit later correction is safe to merge.
    feedback.remove_judgment(record.run_id, key, index_dir=other)
    second = feedback.search_and_record("garden", index_dir=other, use_feedback=False)
    feedback.judge(second.run_id, second.results[0].image_id, "irrelevant", index_dir=other)
    before = [snapshot(path) for path in (source, other)]
    destination = tmp_path / "workspace"
    plan = preview([source, other], destination)
    assert not destination.exists()
    assert plan["searches"] == 2 and plan["events"] == 3 and plan["judgments"] == 1
    progress = []
    imported([source, other], destination, on_progress=progress.append)
    assert before == [snapshot(path) for path in (source, other)]
    with FeedbackStore(destination, readonly=True) as store:
        assert store.run(record.run_id)["results"]
        assert store.judgments(record.run_id) == []
        assert [row["label"] for row in store.history(record.run_id)] == ["relevant", None]
        assert store.memory(store.run(record.run_id)["collection_id"], "red")[0]["label"] is None
    assert collections.list_collections(destination, "personal")[0]["id"] == group["id"]
    assert [
        row.image_id
        for row in search.search_collection("red", index_dir=destination, use_feedback=False)
    ] == [row.image_id for row in record.results]
    assert models_directory(destination) == Path("models").resolve()
    assert {p["phase"] for p in progress} >= {
        "inspecting",
        "copying_artifacts",
        "importing_feedback",
        "publishing",
    }
    assert describe(destination)["workspace_id"]


def test_independent_contradictory_query_judgments_are_not_silently_ordered(library, tmp_path):
    _, source, _, _ = library
    other = tmp_path / "other"
    shutil.copytree(source, other)
    one = feedback.search_and_record("red", index_dir=source, use_feedback=False)
    two = feedback.search_and_record("  RED  ", index_dir=other, use_feedback=False)
    key = one.results[0].image_id
    feedback.judge(one.run_id, key, "relevant", index_dir=source)
    feedback.judge(two.run_id, key, "irrelevant", index_dir=other)
    with pytest.raises(ValueError, match="Independent query judgments conflict"):
        preview([source, other], tmp_path / "workspace")


def test_preview_binds_feedback_and_source_generation(library, tmp_path):
    root, source, _, run = library
    record = feedback.search_and_record("red", index_dir=source, use_feedback=False)
    destination = tmp_path / "workspace"
    plan = preview([source], destination)
    feedback.judge(record.run_id, record.results[0].image_id, "relevant", index_dir=source)
    with pytest.raises(ValueError, match="preview changed"):
        initialize([source], destination, plan["plan_sha256"], models=Path("models"))
    assert not destination.exists()
    other = tmp_path / "other"
    shutil.copytree(source, other)
    from test_collections import image

    image(root / "third.png", "blue")
    run()
    with pytest.raises(ValueError, match="active generation differs"):
        preview([source, other], destination)


@pytest.mark.parametrize("existing", [False, True])
def test_cancellation_and_source_edits_never_publish_partial_workspace(library, tmp_path, existing):
    _, source, _, _ = library
    destination = tmp_path / "workspace"
    if existing:
        (destination / "api-server").mkdir(parents=True)
    before = snapshot(source)

    def cancel(value):
        if value["phase"] == "copying_artifacts":
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        imported([source], destination, on_progress=cancel)
    assert destination.exists() == existing
    assert not (destination / "catalog.sqlite3").exists() and not list(
        tmp_path.glob(".workspace.import-*")
    )
    assert snapshot(source) == before
    record = feedback.search_and_record("red", index_dir=source, use_feedback=False)
    edited = False

    def edit(value):
        nonlocal edited
        if value["phase"] == "copying_artifacts" and not edited:
            edited = True
            feedback.judge(record.run_id, record.results[0].image_id, "relevant", index_dir=source)

    with pytest.raises(ValueError, match="Source changed during import"):
        imported([source], destination, on_progress=edit)
    assert destination.exists() == existing
    assert not (destination / "catalog.sqlite3").exists()


def test_existing_destinations_and_source_nesting_are_refused(library, tmp_path):
    _, source, _, _ = library
    destination = tmp_path / "workspace"
    destination.mkdir()
    (destination / "keep.txt").write_text("preserve")
    for target in (destination, source / "nested", source.parent):
        with pytest.raises(ValueError, match="new destination"):
            preview([source], target)
    assert (destination / "keep.txt").read_text() == "preserve"


def test_duplicate_reviews_are_copied_and_conflicts_rejected(library, tmp_path):
    _, source, _, _ = library
    duplicates.scan(source)
    from realopen_lens.storage.catalog import Catalog
    from realopen_lens.storage.artifacts import load_manifest

    catalog = Catalog(source, readonly=True)
    found, generation = catalog.snapshot("personal")
    data = duplicates.snapshot(source, load_manifest(catalog, generation))
    catalog.close()
    review = source / "duplicates/reviews" / (found["id"] + ".json")
    review.parent.mkdir(parents=True)
    # A retained historical review need not be a current displayed suggestion.
    review.write_text(json.dumps({"historical-pair": "confirmed"}), encoding="utf-8")
    other = tmp_path / "other"
    shutil.copytree(source, other)
    destination = tmp_path / "workspace"
    imported([source, other], destination)
    assert (
        destination / "duplicates/reviews" / review.name
    ).read_text() == '{"historical-pair": "confirmed"}'
    assert (destination / "duplicates" / (data["key"] + ".json")).is_file()
    (other / "duplicates/reviews" / review.name).write_text(
        json.dumps({"historical-pair": "dismissed"})
    )
    with pytest.raises(ValueError, match="Duplicate reviews conflict"):
        preview([source, other], tmp_path / "conflict")


def test_workspace_cli_previews_and_applies_only_the_reviewed_plan(library, tmp_path):
    _, source, _, _ = library
    destination = tmp_path / "workspace"
    args = ["workspace", "import", "--source", str(source), "--index-dir", str(destination)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    result = CliRunner().invoke(cli, [*args, "--apply", plan["plan_sha256"], "--log"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "completed"
    assert '"phase": "copying_artifacts"' in result.stderr


def test_initialization_adopts_only_unused_operational_directory(library, tmp_path):
    from realopen_lens.runtime.jobs import writer_lock

    _, source, _, _ = library
    target = tmp_path / "workspace"
    (target / "api-server").mkdir(parents=True)
    with writer_lock(target / "api-server"):
        plan = preview([source], target)
        assert plan["destination_mode"] == "uninitialized"
        with pytest.raises(ValueError, match="Destination is in use"):
            initialize([source], target, plan["plan_sha256"], models=Path("models"))
    imported([source], target)
    assert (target / "catalog.sqlite3").is_file()
    assert (target / "api-server/indexing.lock").exists()
    with pytest.raises(ValueError, match="new destination"):
        preview([source], target)


def test_invalid_nested_import_does_not_even_create_source_directories(library):
    _, source, _, _ = library
    target = source / "new-parent/workspace"
    with pytest.raises(ValueError, match="outside all source indexes"):
        initialize([source], target, "0" * 64, models=Path("models"))
    assert not target.parent.exists()


def test_conflicting_virtual_revisions_are_not_silently_combined(library, tmp_path):
    _, source, _, _ = library
    group = collections.create(source, "personal", "Saved")
    other = tmp_path / "other"
    shutil.copytree(source, other)
    collections.rename(other, "personal", group["id"], "Renamed", group["revision"])
    with pytest.raises(ValueError, match="Virtual collection revisions"):
        preview([source, other], tmp_path / "workspace")


def test_workspace_info_reports_invalid_metadata_without_traceback(tmp_path):
    (tmp_path / "workspace.json").write_text('{"version":1}', encoding="utf-8")
    result = CliRunner().invoke(cli, ["workspace", "info", "--index-dir", str(tmp_path)])
    assert result.exit_code == 1 and "Invalid workspace metadata" in result.stderr


def test_independent_conflicts_on_retired_pixel_alias_are_detected(library, tmp_path):
    from PIL import Image
    from realopen_lens.models.manager import digest

    root, source, _, run = library
    with Image.open(root / "red.png") as image:
        image.save(root / "red.bmp")
    run()
    other = tmp_path / "other"
    shutil.copytree(source, other)
    first = feedback.search_and_record("red", index_dir=source, use_feedback=False)
    second = feedback.search_and_record("red", index_dir=other, use_feedback=False)
    feedback.judge(first.run_id, digest(root / "red.png"), "relevant", index_dir=source)
    feedback.judge(second.run_id, digest(root / "red.bmp"), "irrelevant", index_dir=other)
    duplicates.scan(source)
    duplicates.scan(other)
    (root / "red.png").unlink()
    duplicates.scan(source)
    duplicates.scan(other)
    with pytest.raises(ValueError, match="Independent query judgments conflict"):
        preview([source, other], tmp_path / "workspace")
