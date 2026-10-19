"""Exact vector search, OCR keyword retrieval, and image-level fusion."""

import math


import re


import sqlite3


import unicodedata


from typing import Any


from collections import defaultdict


import numpy as np


def rank_vectors(
    vectors: np.ndarray,
    query: np.ndarray,
    image_ids: list[str],
    limit: int = 50,
) -> list[str]:
    """Max passage similarity per image; stable ties make reports reproducible."""
    if limit < 1:
        raise ValueError("Retrieval limit must be positive")
    best = vector_scores(vectors, query, image_ids)
    return sorted(best, key=lambda key: (-best[key], key))[:limit]


def lexical_index(records: list[dict[str, Any]]) -> sqlite3.Connection:
    database = sqlite3.connect(":memory:")
    database.execute(
        "CREATE VIRTUAL TABLE ocr USING fts5(image_id UNINDEXED, text, "
        "tokenize=\"unicode61 tokenchars '_'\")"
    )
    contents = []
    for item in records:
        observed = _tokens(item["text"])
        extra = set().union(*(_tokens(text) for text in item.get("alternate_texts", []))) - observed
        contents.append(
            (item["image_id"], _search_text(item["text"]) + "\n" + " ".join(sorted(extra)))
        )
    database.executemany("INSERT INTO ocr VALUES (?, ?)", contents)
    database.execute("CREATE VIRTUAL TABLE ocr_vocab USING fts5vocab(ocr, 'row')")
    return database


def lexical_frequencies(database: sqlite3.Connection, text: str) -> tuple[int, dict[str, int]]:
    terms = sorted(_tokens(text) - _STOPWORDS)
    if len(terms) > 256:
        raise ValueError("Lexical query exceeds 256 distinct terms")
    total = database.execute("SELECT COUNT(*) FROM ocr").fetchone()[0]
    if not terms:
        return total, {}
    placeholders = ",".join("?" for _ in terms)
    frequencies = dict(
        database.execute(f"SELECT term, doc FROM ocr_vocab WHERE term IN ({placeholders})", terms)
    )
    return total, frequencies


def rank_lexical(database: sqlite3.Connection, text: str, limit: int = 50) -> list[str]:
    return list(lexical_scores(database, text, limit))


def lexical_scores(database: sqlite3.Connection, text: str, limit: int = 50) -> dict[str, float]:
    """Retain raw FTS5 BM25 values (lower is better); never mix them with cosine scores."""
    if limit < 1:
        raise ValueError("Retrieval limit must be positive")
    tokens = sorted(_tokens(text))
    if len(set(tokens)) > 256:
        raise ValueError("Lexical query exceeds 256 distinct terms")
    if not tokens:
        return {}
    match = " OR ".join('"' + token + '"' for token in tokens)
    return {
        row[0]: float(row[1])
        for row in database.execute(
            "SELECT image_id, bm25(ocr) FROM ocr WHERE ocr MATCH ? "
            "ORDER BY bm25(ocr), image_id LIMIT ?",
            (match, limit),
        )
    }


def fuse(rankings: list[list[str]], limit: int = 50) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, image_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[image_id] += 1 / (60 + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))[:limit]


def vector_scores(vectors: np.ndarray, query: np.ndarray, image_ids: list[str]) -> dict[str, float]:
    """Keep cosine scores and aggregate passages before combining search channels."""
    if vectors.ndim != 2 or query.ndim != 1 or vectors.shape[1] != query.shape[0]:
        raise ValueError("Embedding dimensions do not match the query")
    if len(vectors) != len(image_ids):
        raise ValueError("Embedding rows do not match image identifiers")
    if not np.isfinite(query).all() or not np.isfinite(vectors).all():
        raise ValueError("Retrieval embeddings must be finite")
    best: dict[str, float] = {}
    for offset in range(0, len(vectors), 4096):
        scores = vectors[offset : offset + 4096] @ query
        for image_id, score in zip(image_ids[offset : offset + 4096], scores, strict=True):
            best[image_id] = max(best.get(image_id, -math.inf), float(score))
    return best


_STOPWORDS = set(
    (
        "a an the of on in at to for with from is are this that and or by "
        "one two some has have there it its as very into next near "
        "le la les un une des du de en au aux et ou avec dans sur est "
        "ce cette ces qui que pour par plus ses son sa "
        "في من على إلى عن هذا هذه هو هي و مع"
    ).split()
)


def _search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text).casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", _search_text(text)))


_STOPWORDS = _tokens(" ".join(_STOPWORDS))
