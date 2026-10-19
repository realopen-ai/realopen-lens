"""OCR character/word error rates against explicitly supplied transcriptions."""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_truth(path: Path | None, image_ids: list[str]) -> list[dict[str, Any]]:
    if path is None:
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("OCR truth must be a JSON list")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) for key in ("image_id", "text", "language")
        ):
            raise ValueError("OCR truth entries need image_id, text, and language strings")
        if not item["language"].strip() or item["image_id"] not in image_ids:
            raise ValueError("OCR truth needs a language and an image in the selected corpus")
        if item["image_id"] in seen:
            raise ValueError(f"Duplicate OCR truth image: {item['image_id']}")
        if "category" in item and not (isinstance(item["category"], str) and item["category"]):
            raise ValueError("OCR category must be a nonempty string")
        phrases = item.get("key_phrases", [])
        if not isinstance(phrases, list) or not all(isinstance(p, str) and p for p in phrases):
            raise ValueError("OCR key_phrases must be a list of nonempty strings")
        if any(normalized(phrase) not in normalized(item["text"]) for phrase in phrases):
            raise ValueError(f"OCR key phrase absent from reference: {item['image_id']}")
        seen.add(item["image_id"])
    return items


def distance(left: list | str, right: list | str) -> int:
    # Myers' exact bit-vector Levenshtein distance. Python's arbitrary-width
    # integers avoid a quadratic Python loop for dense document transcripts.
    if len(left) > len(right):
        left, right = right, left
    length = len(left)
    if not length:
        return len(right)
    matches: dict[Any, int] = {}
    for index, value in enumerate(left):
        matches[value] = matches.get(value, 0) | (1 << index)
    positive, negative, score = ~0, 0, length
    last = 1 << (length - 1)
    for value in right:
        equal = matches.get(value, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        score += bool(plus & last) - bool(minus & last)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = minus | ~(vertical | plus)
        negative = plus & vertical
    return score


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def evaluate(records: list[dict[str, str]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    actual = {record["image_id"]: record["text"] for record in records}
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "images": 0,
            "character_errors": 0,
            "reference_characters": 0,
            "word_errors": 0,
            "reference_words": 0,
            "matched_tokens": 0,
            "reference_tokens": 0,
            "observed_tokens": 0,
            "matched_key_phrases": 0,
            "reference_key_phrases": 0,
        }
    )
    for item in truth:
        expected = normalized(item["text"])
        observed = normalized(actual[item["image_id"]])
        expected_tokens = Counter(re.findall(r"\w+", expected))
        observed_tokens = Counter(re.findall(r"\w+", observed))
        phrases = item.get("key_phrases", [])
        errors = {
            "images": 1,
            "character_errors": distance(expected, observed),
            "reference_characters": len(expected),
            "word_errors": distance(expected.split(), observed.split()),
            "reference_words": len(expected.split()),
            "matched_tokens": sum((expected_tokens & observed_tokens).values()),
            "reference_tokens": expected_tokens.total(),
            "observed_tokens": observed_tokens.total(),
            "matched_key_phrases": sum(normalized(p) in observed for p in phrases),
            "reference_key_phrases": len(phrases),
        }
        for language in {"all", item["language"]}:
            for key, value in errors.items():
                groups[language][key] += value
    result = {}
    for language, counts in groups.items():
        result[language] = {
            **counts,
            "cer": (
                counts["character_errors"] / counts["reference_characters"]
                if counts["reference_characters"]
                else None
            ),
            "wer": (
                counts["word_errors"] / counts["reference_words"]
                if counts["reference_words"]
                else None
            ),
            "token_recall": (
                counts["matched_tokens"] / counts["reference_tokens"]
                if counts["reference_tokens"]
                else None
            ),
            "token_precision": (
                counts["matched_tokens"] / counts["observed_tokens"]
                if counts["observed_tokens"]
                else None
            ),
            "key_phrase_recall": (
                counts["matched_key_phrases"] / counts["reference_key_phrases"]
                if counts["reference_key_phrases"]
                else None
            ),
        }
    return result


def by_category(records: list[dict[str, str]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({item["category"] for item in truth if item.get("category")})
    return {
        category: evaluate(records, [item for item in truth if item.get("category") == category])[
            "all"
        ]
        for category in categories
    }
