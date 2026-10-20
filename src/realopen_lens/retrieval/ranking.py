"""Exact vector search, OCR keyword retrieval, and image-level fusion."""

import math
import re
import sqlite3
import unicodedata
from typing import Any
from collections import defaultdict

import numpy as np

from realopen_lens.retrieval.candidates import CandidateDepths


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


def lexical_evidence(
    query: str, text: str, statistics: tuple[int, dict[str, int]] | None = None
) -> float:
    """Demand substantive token coverage or an exact code identifier/quoted phrase."""
    query = " ".join(unicodedata.normalize("NFKC", query).split())
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    wanted = _tokens(query) - _STOPWORDS
    if not wanted:
        return 0.0
    actual = _tokens(text)
    coverage = len(wanted & actual) / len(wanted)
    identifiers = re.findall(r"\b[\w]+(?:[_./:\\-][\w]+)+\b", query)
    identifiers += re.findall(r"\b[A-Za-z]*[a-z][A-Z]\w*\b", query)
    identifiers += re.findall(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b", query)
    normalized = text.casefold()
    exact_identifier = any(
        re.search(r"(?<!\w)" + re.escape(item.casefold()) + r"(?!\w)", normalized)
        for item in identifiers
    )
    phrases = re.findall(r'"([^"\n]{2,})"', query)
    if exact_identifier or any(
        re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", normalized)
        for phrase in phrases
    ):
        return 1.4
    if statistics is not None:
        total, frequencies = statistics
        searchable = {term for term in wanted if frequencies.get(term, 0) > 0}
        matched = searchable & actual
        if len(matched) < 2:
            return 0.0
        weights = {term: 1 + math.log((total + 1) / (frequencies[term] + 1)) for term in searchable}
        coverage = sum(weights[term] for term in matched) / sum(weights.values())
        if any(frequencies[term] <= max(2, total * 0.01) for term in matched):
            return max(0.8, coverage)
        # Common administrative vocabulary is weak text evidence, even if it
        # covers most of a photo caption's indexable words.
        return coverage * 0.5 if coverage >= 0.6 else 0.0
    if len(wanted) < 2 or coverage < 0.6:
        return 0.0
    return coverage


def hybrid_rank(
    visual_vectors: np.ndarray,
    visual_query: np.ndarray,
    image_ids: list[str],
    passage_vectors: np.ndarray,
    semantic_query: np.ndarray,
    passage_ids: list[str],
    query: str,
    ocr_records: list[dict[str, Any]],
    limit: int = 50,
    semantic_floor: float = 0.83,
    semantic_margin: float = 0.04,
    lexical_candidates: list[str] | None = None,
    lexical_statistics: tuple[int, dict[str, int]] | None = None,
    trace: dict | None = None,
    candidate_depths: CandidateDepths | None = None,
) -> list[str]:
    """Score-preserving fusion with one bounded OCR contribution per image.

    Weak OCR evidence leaves visual ordering unchanged. Lexical and semantic OCR
    are correlated observations of the same content and must not get two votes.
    Semantic thresholds are experimental defaults tied to the pinned E5 export;
    reports must validate them on held-out data before model changes.
    """
    if limit < 1:
        raise ValueError("Retrieval limit must be positive")
    depths = candidate_depths or CandidateDepths(visual=max(50, limit))
    visual = vector_scores(visual_vectors, visual_query, image_ids)
    semantic = (
        vector_scores(passage_vectors, semantic_query, passage_ids) if len(passage_ids) else {}
    )
    background = float(np.median(list(semantic.values()))) if semantic else 0
    texts = {row["image_id"]: row["text"] for row in ocr_records}
    alternatives = {row["image_id"]: row.get("alternate_texts", []) for row in ocr_records}
    if lexical_candidates is None or lexical_statistics is None:
        database = lexical_index(ocr_records)
        try:
            if lexical_candidates is None:
                lexical_candidates = rank_lexical(database, query, depths.lexical)
            if lexical_statistics is None:
                lexical_statistics = lexical_frequencies(database, query)
        finally:
            database.close()
    visual_ranking = sorted(visual, key=lambda key: (-visual[key], key))[: depths.visual]
    semantic_ranking = sorted(semantic, key=lambda key: (-semantic[key], key))[: depths.semantic]
    lexical_ids = set(lexical_candidates[: depths.lexical])
    candidates = set(visual_ranking)
    candidates.update(semantic_ranking)
    candidates.update(lexical_ids)
    combined = {}
    evidence = {}
    for image_id in candidates & visual.keys():
        score = visual[image_id]
        lexical = (
            lexical_evidence(query, texts.get(image_id, ""), lexical_statistics)
            if image_id in lexical_ids
            else 0
        )
        if image_id in lexical_ids:
            lexical = max(
                lexical,
                max(
                    (
                        0.75 * lexical_evidence(query, text, lexical_statistics)
                        for text in alternatives.get(image_id, [])
                    ),
                    default=0,
                ),
            )
        similarity = semantic.get(image_id, 0)
        strength = max(
            0,
            min(
                1,
                (similarity - semantic_floor) / 0.06,
                (similarity - background - semantic_margin) / 0.06,
            ),
        )
        combined[image_id] = score + 0.16 * max(lexical, strength)
        if trace is not None:
            evidence[image_id] = {
                "visual_cosine": score,
                "semantic_cosine": semantic.get(image_id),
                "lexical_evidence": lexical,
                "semantic_strength": strength,
                "hybrid_score": combined[image_id],
            }
    ordered = sorted(combined, key=lambda key: (-combined[key], key))
    if trace is not None:
        trace.update(
            {
                "candidates": [{"image_id": key, **evidence[key]} for key in ordered],
                "candidate_depths": depths.as_dict(),
                "rankings": {
                    "visual": visual_ranking,
                    "semantic": semantic_ranking,
                    "lexical": lexical_candidates[: depths.lexical],
                    "hybrid": ordered[:50],
                },
            }
        )
    return ordered[:limit]
