import json

import pytest

from benchmarks.judgments import queries
from realopen_lens.corpus import scan


def test_scan_is_sorted_bounded_and_excludes_symlinks_outside_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    for name in ["b.JPG", "a.png", "notes.txt"]:
        (root / name).touch()
    outside = tmp_path / "private.png"
    outside.touch()
    try:
        (root / "escape.png").symlink_to(outside)
    except OSError:
        pass  # Windows may require elevated privileges to create symlinks.
    assert scan(root, 1) == ["a.png"]
    assert scan(root, 10) == ["a.png", "b.JPG"]


def test_empty_data_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="Add images to data/"):
        scan(tmp_path, 100)


@pytest.mark.parametrize(
    "changes",
    [{"relevant": ["missing.png"]}, {"language": ""}, {"relevant": "a.png"}, {"text": 123}],
)
def test_queries_reject_invalid_judgments(tmp_path, changes):
    path = tmp_path / "queries.json"
    item = {"id": "q", "text": "a diagram", "language": "en", "relevant": ["a.png"]}
    path.write_text(json.dumps([{**item, **changes}]))
    with pytest.raises(ValueError):
        queries(path, ["a.png"])


def test_query_ids_must_be_unique(tmp_path):
    path = tmp_path / "queries.json"
    item = {"id": "q", "text": "a diagram", "language": "en"}
    path.write_text(json.dumps([item, item]))
    with pytest.raises(ValueError, match="Duplicate query"):
        queries(path, ["a.png"])
    assert queries(None, ["a.png"]) == []
