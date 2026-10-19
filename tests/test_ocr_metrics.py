import pytest

from benchmarks.metrics.ocr import by_category, evaluate, load_truth


def test_ocr_error_rates_include_missing_and_extra_words():
    truth = [{"image_id": "a", "text": "one two", "language": "en"}]
    scores = evaluate([{"image_id": "a", "text": "one two extra"}], truth)
    assert scores["en"]["wer"] == 0.5
    assert scores["all"]["images"] == 1
    assert evaluate([{"image_id": "a", "text": ""}], truth)["en"]["cer"] == 1


def test_ocr_normalization_and_unicode():
    truth = [
        {"image_id": "a", "text": "CAFÉ\n facture", "language": "fr"},
        {"image_id": "b", "text": "الاجتماع", "language": "ar"},
    ]
    scores = evaluate(
        [{"image_id": "a", "text": "cafe\u0301  facture"}, {"image_id": "b", "text": "الاجتماع"}],
        truth,
    )
    assert scores["all"]["cer"] == pytest.approx(0)
    assert scores["ar"]["wer"] == 0


def test_no_reference_is_not_reported_as_perfect_ocr():
    assert evaluate([], []) == {}
    scores = evaluate(
        [{"image_id": "a", "text": "noise"}], [{"image_id": "a", "text": "", "language": "en"}]
    )
    assert scores["all"]["cer"] is None
    assert scores["all"]["character_errors"] == 5


def test_token_recovery_ignores_reading_order_but_counts_duplicates():
    records = [{"image_id": "a", "text": "Database API API"}]
    truth = [
        {
            "image_id": "a",
            "language": "en",
            "category": "diagram",
            "text": "API API Database",
            "key_phrases": ["API", "Database"],
        }
    ]
    score = evaluate(records, truth)["all"]
    assert score["wer"] > 0
    assert score["token_recall"] == 1
    assert score["token_precision"] == 1
    assert score["key_phrase_recall"] == 1
    records[0]["text"] = "API Database"
    assert evaluate(records, truth)["all"]["token_recall"] == pytest.approx(2 / 3)
    assert by_category(records, truth)["diagram"]["images"] == 1


def test_identifier_matching_preserves_punctuation():
    records = [{"image_id": "a", "text": "show locals=True ZeroDivisionError"}]
    truth = [
        {
            "image_id": "a",
            "language": "en",
            "text": "show_locals=True ZeroDivisionError",
            "key_phrases": ["show_locals", "ZeroDivisionError"],
        }
    ]
    assert evaluate(records, truth)["all"]["key_phrase_recall"] == 0.5


def test_ground_truth_cannot_claim_a_key_phrase_it_does_not_contain(tmp_path):
    import json

    path = tmp_path / "truth.json"
    path.write_text(
        json.dumps(
            [{"image_id": "a", "language": "en", "text": "hello", "key_phrases": ["goodbye"]}]
        )
    )
    with pytest.raises(ValueError, match="absent from reference"):
        load_truth(path, ["a"])
