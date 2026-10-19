"""OCR quality verdicts and exact-identifier preservation diagnostics."""

import re
from benchmarks.metrics.ocr import evaluate, normalized


def image_quality(row: dict, text: str) -> dict:
    if "truth" in row:
        return evaluate(
            [{"image_id": row["id"], "text": text}],
            [
                {
                    "image_id": row["id"],
                    "text": row["truth"],
                    "language": row.get("language", "en"),
                    "key_phrases": row.get("key_phrases", []),
                }
            ],
        )["all"]
    phrases = row.get("key_phrases", [])
    if not phrases:
        return {}
    empty = evaluate(
        [{"image_id": row["id"], "text": ""}],
        [{"image_id": row["id"], "text": "", "language": "en"}],
    )["all"]
    return {
        **empty,
        "matched_key_phrases": sum(normalized(p) in normalized(text) for p in phrases),
        "reference_key_phrases": len(phrases),
        "key_phrase_recall": sum(normalized(p) in normalized(text) for p in phrases) / len(phrases),
    }


def quality_status(quality: dict) -> str:
    if not quality:
        return "not_evaluated"
    checks = [
        quality[key] >= 0.8
        for key in ("token_recall", "token_precision", "key_phrase_recall")
        if quality.get(key) is not None
    ]
    return "pass" if checks and all(checks) else "fail"


def identifier_preservation(identifiers: list[str], text: str) -> dict:
    """Exact, case-sensitive identifier matches; token recall can hide punctuation loss."""
    return {
        identifier: bool(re.search(r"(?<!\w)" + re.escape(identifier) + r"(?!\w)", text))
        for identifier in identifiers
    }
