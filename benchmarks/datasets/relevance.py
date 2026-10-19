"""Validate frozen, explicitly reviewed query/document relevance and leakage boundaries."""

import json
from pathlib import Path
import unicodedata

from benchmarks.datasets.downloads import digest

SPLITS = {"train", "validation", "test"}


def load(corpus: Path, judgments: Path) -> tuple[dict, dict]:
    manifest = json.loads(corpus.read_text(encoding="utf-8"))
    bundle = json.loads(judgments.read_text(encoding="utf-8"))
    if bundle.get("version") != 1 or bundle.get("corpus_sha256") != digest(corpus):
        raise ValueError("Judgments do not identify this frozen corpus")
    images: dict[str, dict] = {}
    boundaries: dict[tuple[str, str], str] = {}

    def boundary(kind, key, split):
        if not isinstance(key, str) or not key.strip() or split not in SPLITS:
            raise ValueError("Invalid group or split")
        group = (kind, key)
        if group in boundaries and boundaries[group] != split:
            raise ValueError(f"Split leakage in {kind}: {key}")
        boundaries[group] = split

    for row in manifest["images"]:
        key = row["sha256"]
        if key in images or row["id"] in {r["id"] for r in images.values()}:
            raise ValueError("Duplicate corpus image identity")
        images[key] = row
        for field in ("source_document", "duplicate_group", "pixel_sha256"):
            boundary(field, row[field], row["split"])
    seen: set[str] = set()
    texts: dict[str, str] = {}
    for query in bundle["queries"]:
        if query["id"] in seen:
            raise ValueError("Duplicate query identity")
        seen.add(query["id"])
        if not all(
            isinstance(query.get(k), str) and query[k].strip()
            for k in ("id", "text", "language", "family")
        ):
            raise ValueError("Missing query identity, language, or family")
        if len(query["text"]) > 4096:
            raise ValueError("Query is too long")
        boundary("query_family", query["family"], query["split"])
        normalized = " ".join(unicodedata.normalize("NFKC", query["text"]).casefold().split())
        if normalized in texts and texts[normalized] != query["family"]:
            raise ValueError("Identical query text crosses families")
        texts[normalized] = query["family"]
        review = query.get("review", {})
        if review.get("status") != "reviewed" or not all(
            review.get(k) for k in ("reviewer", "method", "date")
        ):
            raise ValueError("Queries must carry explicit review provenance")
        labels = query["judgments"]
        if not labels or not any(r["grade"] > 0 for r in labels):
            raise ValueError("Each evaluated query needs a reviewed positive")
        judged = set()
        for label in labels:
            key = label["image_sha256"]
            if key not in images or key in judged:
                raise ValueError("Missing or duplicate judged image")
            if type(label["grade"]) is not int or label["grade"] not in (0, 1, 2):
                raise ValueError("Relevance grade must be 0, 1, or 2")
            if not label.get("reason"):
                raise ValueError("Judgments need a review reason")
            judged.add(key)
            if images[key]["split"] != query["split"]:
                raise ValueError("Judged document and query must share a split")
    if not seen:
        raise ValueError("No reviewed queries")
    return manifest, bundle
