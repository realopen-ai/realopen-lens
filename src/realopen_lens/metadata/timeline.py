"""Model-free gallery chronology over immutable indexed metadata and signatures."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from realopen_lens.storage.organization import CollectionConflict

ORDERS = {"newest", "oldest"}


def image_date(row: dict, paths: tuple[str, ...] | list[str], root: Path) -> dict:
    captured = row.get("metadata", {}).get("exif", {}).get("captured_at")
    if isinstance(captured, str):
        try:
            value = datetime.fromisoformat(captured)
            return {
                "date": value.isoformat(),
                "day": value.date().isoformat(),
                "month": value.date().isoformat()[:7],
                "source": "capture",
                "timezone_known": value.utcoffset() is not None,
            }
        except ValueError:
            pass
    values = []
    available = set(paths)
    for relative, signature in row.get("signatures", {}).items():
        if str((root / relative).resolve()) not in available:
            continue
        try:
            stamp = signature["mtime_ns"]
            if not isinstance(stamp, int) or isinstance(stamp, bool):
                continue
            values.append(datetime.fromtimestamp(stamp / 1e9, timezone.utc))
        except (KeyError, ValueError, OverflowError, OSError):
            continue
    if values:
        value = max(values)
        return {
            "date": value.isoformat(),
            "day": value.date().isoformat(),
            "month": value.date().isoformat()[:7],
            "source": "modified",
            "timezone_known": True,
        }
    return {"date": None, "day": None, "month": None, "source": "unknown"}


def date_key(value: dict) -> str:
    # Capture calendar dates stay as recorded, including when their offset is unknown.
    return (
        datetime.fromisoformat(value["date"])
        .replace(tzinfo=None)
        .isoformat(timespec="microseconds")
        if value.get("date")
        else ""
    )


def preferred_date(values: list[dict]) -> dict:
    """Copies use capture metadata first, then the newest recorded modification."""
    return max(values, key=lambda value: (value["source"] == "capture", date_key(value)))


def ordered(dates: dict[str, dict], order: str) -> list[str]:
    if order not in ORDERS:
        raise ValueError("Use newest or oldest timeline order")
    known = sorted(key for key, value in dates.items() if value["date"])
    known.sort(key=lambda key: date_key(dates[key]), reverse=order == "newest")
    return known + sorted(key for key, value in dates.items() if not value["date"])


def snapshot(identity: object, dates: dict[str, dict], order: str, expected: str | None) -> str:
    token = hashlib.sha256(
        json.dumps([identity, order, dates], sort_keys=True).encode("utf-8")
    ).hexdigest()
    if expected is not None and token != expected:
        raise CollectionConflict("Image timeline changed; reload images to continue")
    return token


def page(records: list[dict], dates: dict[str, dict], offset: int, total: int) -> dict:
    return {
        "results": [
            {**record, "rank": offset + i + 1, "timeline": dates[record["image_id"]]}
            for i, record in enumerate(records)
        ],
        "next_offset": offset + len(records) if offset + len(records) < total else None,
    }
