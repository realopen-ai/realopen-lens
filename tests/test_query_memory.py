"""Query memory is explicit, scoped, reversible and independent of baseline ranking."""

from copy import deepcopy
import json
import sqlite3

import pytest
from typer.testing import CliRunner

from test_feedback import saved as saved_fixture
from test_virtual_collections import library as library_fixture
from test_collections import app as collection_fixture
from realopen_lens.cli import app as cli
from realopen_lens.learning import feedback
from realopen_lens.storage.feedback import FeedbackStore, FeedbackConflict, normalize_query
from realopen_lens.retrieval import search, memory
from realopen_lens.organization import collections, duplicates
from realopen_lens.models.manager import digest

saved = saved_fixture
library = library_fixture
app = collection_fixture


def test_normalization_is_conservative_and_latest_action_including_clear_wins(saved):
    root, old, payload = saved
    assert normalize_query("  ＣＡＦÉ\n  ERR_42 ") == "café err_42"
    assert normalize_query("cafe") != normalize_query("café")
    assert normalize_query("ERR-42") != normalize_query("ERR_42")
    first = feedback.change_judgment(old, "one", "relevant", index_dir=root)["memory"]
    newer = deepcopy(payload)
    newer["query"]["text"] = "  فاتورة   CAFÉ "
    with FeedbackStore(root) as store:
        newer.update(run_id=old, created_at="old payload timestamp")
        new = store.add_run(newer)
        assert store.run(new)["run_id"] == new
        assert store.run(new)["created_at"] != "old payload timestamp"
    second = feedback.change_judgment(
        new, "one", "irrelevant", index_dir=root, expected_event_id=first["event_id"]
    )["memory"]
    with pytest.raises(FeedbackConflict):
        feedback.change_judgment(
            old, "one", None, index_dir=root, expected_event_id=first["event_id"]
        )
    # Repeating an old run's prior label is a new explicit action.
    third = feedback.change_judgment(old, "one", "relevant", index_dir=root)["memory"]
    assert third["event_id"] > second["event_id"]
    cleared = feedback.change_judgment(new, "one", None, index_dir=root)["memory"]
    with FeedbackStore(root, readonly=True) as store:
        assert store.memory("collection", "فاتورة café")[0] == cleared
        assert cleared["label"] is None
        assert store.judgments(old)[0]["label"] == "relevant"  # History retained.
    output = root / "labels.jsonl"
    feedback.export_feedback(output, index_dir=root)
    assert json.loads(output.read_text(encoding="utf-8"))["superseded"] is True


def test_legacy_read_only_fallback_and_transactional_migration_preserve_events(saved):
    root, run, _ = saved
    feedback.judge(run, "one", "irrelevant", index_dir=root)
    feedback.remove_judgment(run, "one", index_dir=root)
    # Reconstruct the existing v1 schema without projection tables.
    with sqlite3.connect(root / "feedback.sqlite3") as db:
        db.execute("DROP TABLE query_memory")
        db.execute("DROP TABLE memory_runs")
        db.execute("PRAGMA user_version=1")
    before = (root / "feedback.sqlite3").read_bytes()
    with FeedbackStore(root, readonly=True) as store:
        expected = store.memory("collection", "فاتورة café")
        assert store.version == 1 and expected[0]["label"] is None
    assert (root / "feedback.sqlite3").read_bytes() == before
    with FeedbackStore(root) as store:
        assert store.version == 2 and store.memory("collection", "فاتورة café") == expected
        assert len(store.history(run)) == 2


def test_memory_key_isolates_sources_scopes_and_image_queries(saved):
    root, run, payload = saved
    feedback.judge(run, "one", "irrelevant", index_dir=root)
    with FeedbackStore(root) as store:
        for changes in (
            {"collection_id": "other"},
            {"scope": {"id": "virtual"}},
            {"query": {"kind": "image", "image_id": "one"}},
        ):
            other = store.add_run({**payload, **changes})
            store.judge(other, "one", "relevant")
        assert store.memory("collection", "فاتورة café")[0]["label"] == "irrelevant"
        assert store.memory("collection", "فاتورة café", "virtual")[0]["label"] == "relevant"
        assert store.memory("other", "فاتورة café")[0]["label"] == "relevant"
        assert store.memory("collection", "فاتورة cafe") == []


@pytest.mark.parametrize("mode", ["hybrid", "visual", "semantic", "lexical"])
def test_production_memory_promotes_hides_backfills_and_baseline_is_unchanged(library, mode):
    root, index, _, _ = library
    # Both documents match the lexical query.
    query = "red OR garden"
    baseline_memory = {}
    options = {"index_dir": index, "mode": mode, "limit": 2}
    baseline = search.search_collection(query, use_feedback=False, **options)
    saved_run = feedback.search_and_record(query, use_feedback=False, **options)
    first, second = (row.image_id for row in baseline)
    feedback.judge(saved_run.run_id, second, "relevant", index_dir=index)
    promoted = search.search_collection(query, feedback_trace=baseline_memory, **options)
    assert promoted[0].image_id == second
    assert promoted[0].feedback["label"] == "relevant"
    assert baseline_memory["applied"] and baseline_memory["positive_groups"] == 1
    feedback.judge(saved_run.run_id, first, "irrelevant", index_dir=index)
    assert [row.image_id for row in search.search_collection(query, **options)] == [second]
    assert search.search_collection(query, use_feedback=False, **options) == baseline
    hidden = memory.hidden(index, query, limit=1)
    assert hidden["total"] == 1 and hidden["results"][0]["image_id"] == first
    assert memory.hidden(index, query, offset=1)["results"] == []
    assert (
        CliRunner().invoke(cli, ["feedback", "hidden", query, "--index-dir", str(index)]).exit_code
        == 0
    )
    # A hidden stale/removed source never reappears through remembered positives.
    (root / ("red.png" if second == digest(root / "red.png") else "green.png")).unlink()
    assert search.search_collection(query, **options) == []


def test_remembered_positive_outside_candidate_pool_is_recovered_without_score_offsets(
    library, monkeypatch
):
    root, index, _, _ = library
    recorded = feedback.search_and_record("red", index_dir=index, use_feedback=False, limit=2)
    positive = digest(root / "green.png")
    feedback.judge(recorded.run_id, positive, "relevant", index_dir=index)
    monkeypatch.setattr(search, "hybrid_rank", lambda *a, **kw: [digest(root / "red.png")])
    results = search.search_collection("red", index_dir=index, limit=1)
    assert results[0].image_id == positive and results[0].feedback["label"] == "relevant"


def test_virtual_memory_stays_in_scope_and_follows_renames(library):
    root, index, _, _ = library
    red = digest(root / "red.png")
    group = collections.create(index, "personal", "Reviewed")
    group = collections.change_members(index, "personal", group["id"], [red], group["revision"])
    recorded = feedback.search_and_record("red", index_dir=index, virtual_collection=group["id"])
    feedback.judge(recorded.run_id, red, "irrelevant", index_dir=index)
    assert search.search_collection("red", index_dir=index, virtual_collection=group["id"]) == []
    assert search.search_collection("red", index_dir=index)
    assert memory.hidden(index, "red", virtual_collection=group["id"])["total"] == 1
    group = collections.rename(index, "personal", group["id"], "Renamed", group["revision"])
    assert memory.hidden(index, "red", virtual_collection=group["id"])["total"] == 1
    collections.change_members(
        index, "personal", group["id"], [red], group["revision"], remove=True
    )
    assert memory.hidden(index, "red", virtual_collection=group["id"])["total"] == 0


def test_pixel_aliases_share_latest_action_but_near_duplicates_do_not(library):
    from PIL import Image

    root, index, _, run = library
    with Image.open(root / "red.png") as source:
        source.save(root / "red.bmp")
        source.resize((32, 24)).save(root / "small.png")
    run()
    duplicates.scan(index, "personal")
    recorded = feedback.search_and_record("red", index_dir=index, limit=10)
    group = next(row for row in recorded.results if len(row.duplicate_ids) == 1)
    decision = feedback.change_judgment(
        recorded.run_id, group.image_id, "irrelevant", index_dir=index
    )["memory"]
    results = search.search_collection("red", index_dir=index)
    assert not {group.image_id, *group.duplicate_ids} & {row.image_id for row in results}
    assert digest(root / "small.png") in {row.image_id for row in results}
    assert memory.hidden(index, "red")["total"] == 1
    feedback.change_judgment(
        recorded.run_id,
        group.image_id,
        None,
        index_dir=index,
        expected_event_id=decision["event_id"],
    )
    assert len(search.search_collection("red", index_dir=index)) == 3


def test_retired_exact_encoding_keeps_memory_and_cas_on_remaining_alias(library):
    from PIL import Image

    root, index, _, run = library
    with Image.open(root / "red.png") as source:
        source.save(root / "red.bmp")
    run()
    # Record both byte identities before publishing decoded-pixel groups.
    historical = feedback.search_and_record("red", index_dir=index, use_feedback=False)
    retired, remaining = digest(root / "red.png"), digest(root / "red.bmp")
    first = feedback.change_judgment(historical.run_id, retired, "irrelevant", index_dir=index)[
        "memory"
    ]
    duplicates.scan(index)
    (root / "red.png").unlink()
    run()
    duplicates.scan(index)
    assert remaining not in [
        row.image_id for row in search.search_collection("red", index_dir=index)
    ]
    page = memory.hidden(index, "red")
    assert page["total"] == 1 and page["results"][0]["image_id"] == remaining
    fresh = feedback.search_and_record("red", index_dir=index, use_feedback=False)
    feedback.change_judgment(
        fresh.run_id, remaining, "relevant", index_dir=index, expected_event_id=first["event_id"]
    )
    assert search.search_collection("red", index_dir=index)[0].image_id == remaining


def test_two_stale_corrections_cannot_both_overwrite_latest_query_decision(saved):
    from concurrent.futures import ThreadPoolExecutor

    root, run, payload = saved
    first = feedback.change_judgment(run, "one", "relevant", index_dir=root)["memory"]
    with FeedbackStore(root) as store:
        newer = store.add_run(payload)

    def change(run_id):
        try:
            return feedback.change_judgment(
                run_id, "one", "irrelevant", index_dir=root, expected_event_id=first["event_id"]
            )
        except FeedbackConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(change, [run, newer]))
    assert sum(result is not None for result in results) == 1
