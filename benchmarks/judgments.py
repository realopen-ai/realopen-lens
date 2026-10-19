"""Read and validate benchmark query relevance judgments."""

import json
from pathlib import Path
from typing import Any


def queries(path: Path | None, image_ids: list[str]) -> list[dict[str, Any]]:
    if path is None:
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Queries must be a JSON list")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each query must be an object")
        if not all(
            isinstance(item.get(k), str) and item[k].strip() for k in ("id", "text", "language")
        ):
            raise ValueError("Each query needs nonempty id, text, and language strings")
        if item["id"] in seen:
            raise ValueError(f"Duplicate query id: {item['id']}")
        seen.add(item["id"])
        relevant = item.get("relevant", [])
        if not isinstance(relevant, list) or not all(isinstance(x, str) for x in relevant):
            raise ValueError("Query relevant must be a list of image paths relative to the corpus")
        missing = set(relevant) - set(image_ids)
        if missing:
            raise ValueError(f"Judged images absent from selected corpus: {sorted(missing)}")
        item["relevant"] = relevant
    return items
