"""Frozen pre-change intent cases and conservative ranking activation checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from realopen_lens.retrieval.ranking import hybrid_rank, lexical_evidence
from realopen_lens.retrieval.identifiers import numeric_identifier_match

CASES = json.loads(
    (
        Path(__file__).resolve().parents[1] / "benchmarks/sources/numeric-identifiers-v1/cases.json"
    ).read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_frozen_numeric_intent_cases(case):
    assert numeric_identifier_match(case["query"], case["text"]) is case["expected_numeric_match"]
    if "existing_exact_code" in case:
        assert (lexical_evidence(case["query"], case["text"], (100, {})) == 1.4) is case[
            "existing_exact_code"
        ]


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["expected_numeric_match"]],
    ids=[c["id"] for c in CASES if c["expected_numeric_match"]],
)
def test_strong_typed_numeric_match_can_recover_weak_visual_target(case):
    trace = {}
    results = hybrid_rank(
        np.array([[0.45], [0.30]], dtype=np.float32),
        np.ones(1),
        ["visual", "target"],
        np.empty((0, 1), dtype=np.float32),
        np.ones(1),
        [],
        case["query"],
        [
            {"image_id": "visual", "text": "Unrelated landscape"},
            {"image_id": "target", "text": case["text"]},
        ],
        limit=2,
        trace=trace,
    )
    assert results[0] == "target"
    target = next(r for r in trace["candidates"] if r["image_id"] == "target")
    assert target["lexical_evidence"] == 1.4


def test_unsupported_single_number_does_not_get_identifier_strength_from_rarity():
    assert not numeric_identifier_match("4868", "Form 4868")
    assert lexical_evidence("4868", "Form 4868", (10000, {"4868": 1})) == 0


@pytest.mark.parametrize("value", ["2026-10-20", "2026/10/20", "100.00"])
def test_punctuation_alone_does_not_turn_numbers_into_strong_code_evidence(value):
    assert lexical_evidence(value, "Date or amount " + value) < 1.4


def test_explicit_quoted_dates_and_valid_ip_addresses_keep_literal_evidence():
    assert lexical_evidence('"2026-10-20"', "Date 2026-10-20") == 1.4
    assert lexical_evidence("192.168.1.42", "Host 192.168.1.42") == 1.4


@pytest.mark.parametrize("query", ["Form 4868", "نموذج ٤٨٦٨", "Formulaire ۴۸۶۸"])
def test_query_aliases_find_existing_digit_scripts_without_reindexing(query):
    from realopen_lens.retrieval.ranking import lexical_index, rank_lexical

    database = lexical_index(
        [
            {"image_id": "ascii", "text": "Form 4868"},
            {"image_id": "arabic", "text": "نموذج ٤٨٦٨"},
            {"image_id": "persian", "text": "Formulaire ۴۸۶۸"},
            {"image_id": "wrong", "text": "Form 48680"},
        ]
    )
    try:
        found = rank_lexical(database, query)
        assert {"ascii", "arabic", "persian"} <= set(found)
        assert not numeric_identifier_match(query, "Form 48680")
    finally:
        database.close()


def test_document_generator_preserves_inputs_and_records_content_hashes(tmp_path):
    from benchmarks.datasets.identifier_fixtures import build
    from benchmarks.datasets.downloads import digest

    manifest = build(tmp_path)
    assert len(manifest["documents"]) == 14
    assert all(digest(tmp_path / r["path"]) == r["sha256"] for r in manifest["documents"])
    assert build(tmp_path) == manifest
    first = tmp_path / manifest["documents"][0]["path"]
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Refusing"):
        build(tmp_path)


@pytest.mark.parametrize("endpoint", ["192.168.116.6:2222", "127.0.0.1:8080"])
def test_valid_ip_endpoints_keep_existing_code_strength(endpoint):
    assert lexical_evidence(endpoint, "Server " + endpoint) == 1.4


def test_document_generator_records_platform_rendering_variants(tmp_path, monkeypatch):
    from benchmarks.datasets import identifier_fixtures

    original = identifier_fixtures.ImageFont.load_default
    monkeypatch.setattr(
        identifier_fixtures.ImageFont, "load_default", lambda size: original(size=size + 1)
    )
    result = identifier_fixtures.build(tmp_path)
    assert all(not row["matches_reference_bytes"] for row in result["documents"])
    assert all(len(row["sha256"]) == 64 for row in result["documents"])


@pytest.mark.parametrize(
    "query,text",
    [
        ("Form 1099-MISC", "Form 1099-NEC"),
        ("Form 1099-NEC", "Form 1099-MISC"),
        ("Form 1040", "Form 1040-SR"),
        ("Form 1040", "Form 8962\nAttach to Form 1040, 1040-SR or 1040-NR"),
        ("Form 1099-MISC", "Form 4506-T\nRequest a transcript for Form 1099"),
        ("invoice INV-004231", "Invoice 004231"),
        ("Form 1040", "Form 1040NR\nSee Form 1040 for details"),
    ],
)
def test_compound_identifiers_and_cross_references_do_not_get_partial_numeric_boost(query, text):
    assert not numeric_identifier_match(query, text)


def test_qualifying_first_identifier_can_follow_misread_header_without_fuzzy_repair():
    assert numeric_identifier_match(
        "نموذج 4868", "rom 4868\nGo to www.irs.gov/Form4868\nFile Form 4868"
    )
    assert numeric_identifier_match("Error:1045", "Error:1045 Access denied")


@pytest.mark.parametrize(
    "query",
    [
        "invoice for 2026",
        "فاتورة لعام 2026",
        "facture pour 2026",
        "error after processing 1045 rows",
        "form from 2026",
    ],
)
def test_unbound_date_or_quantity_is_not_an_explicit_identifier(query):
    text = "Invoice 2026\nError 1045\nForm 2026"
    assert not numeric_identifier_match(query, text)
    assert lexical_evidence(query, text) < 1.4
