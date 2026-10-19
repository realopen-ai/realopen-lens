import pytest


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
