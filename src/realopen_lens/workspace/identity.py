"""Describe the running package and workspace without requiring Git or network access."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from realopen_lens import __version__
from realopen_lens.storage.catalog import default_directory


def fingerprint(root: Path, pattern: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob(pattern)):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def build() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {
        "version": __version__,
        "code_sha256": fingerprint(root, "*.py"),
        "gallery_sha256": fingerprint(root / "api/static", "*"),
        "package_path": str(root),
        "gallery_available": (root / "api/static/index.html").is_file(),
    }


def metadata(root: Path) -> dict | None:
    path = root / "workspace.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in ("id", "created_at", "models_directory")
        )
    ):
        raise ValueError(
            "Invalid workspace metadata; restore workspace.json from its import record"
        )
    if not Path(value["models_directory"]).is_absolute():
        raise ValueError("Workspace model directory must be absolute")
    return value


def models_directory(root: Path) -> Path:
    value = metadata(root)
    if value:
        return Path(value["models_directory"])
    return Path("models").resolve()


def describe(root: Path, models: Path | None = None, *, running_build: dict | None = None) -> dict:
    root = root.expanduser().resolve()
    active = running_build or build()
    checkout = Path.cwd() / "src/realopen_lens"
    comparison = None
    if (checkout / "workspace/identity.py").is_file():
        comparison = {
            "code_matches": fingerprint(checkout, "*.py") == active["code_sha256"],
            "gallery_matches": fingerprint(checkout / "api/static", "*")
            == active["gallery_sha256"],
        }
    stored = metadata(root)
    return {
        "directory": str(root),
        "default_directory": str(default_directory()),
        "is_default": root == default_directory().resolve(),
        "initialized": (root / "catalog.sqlite3").is_file(),
        "models_directory": str((models or models_directory(root)).resolve()),
        "workspace_id": stored.get("id") if stored else None,
        "created_at": stored.get("created_at") if stored else None,
        "build": active,
        "checkout": comparison,
    }
