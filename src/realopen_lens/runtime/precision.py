"""Shared device defaults; explicit choices and saved index profiles take priority."""

import json
from pathlib import Path

import psutil

from realopen_lens.storage.catalog import default_directory


def memory_bytes() -> int | None:
    try:
        return psutil.virtual_memory().total
    except (OSError, NotImplementedError, psutil.Error):
        return None


def recommended_precision(total: int | None) -> str:
    """Use physical RAM, never fluctuating free memory; unknown devices use INT8."""
    # Preserve the launcher's decimal threshold for the nominal 16 GB class.
    return "fp32" if total is not None and total >= 16_000_000_000 else "int8"


def default_precision(explicit: str | None = None, *, workspace: Path | None = None) -> str:
    """Resolve an explicit override, saved workspace preference, then device RAM."""
    if explicit is not None:
        if explicit not in {"int8", "fp32"}:
            raise ValueError("SigLIP2 precision must be int8 or fp32")
        return explicit
    settings = (workspace or default_directory()) / "desktop.json"
    if settings.is_file():
        try:
            config = json.loads(settings.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise ValueError("Invalid saved workspace precision settings") from error
        if not isinstance(config, dict):
            raise ValueError("Invalid saved workspace precision settings")
        if "siglip_precision" in config:
            selected = config["siglip_precision"]
            if selected not in ("int8", "fp32"):
                raise ValueError("Invalid saved workspace precision")
            return selected
    return recommended_precision(memory_bytes())
