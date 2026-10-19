"""Regression cases for observed preprocessing, OCR, fusion, and resume failures."""

import random


import subprocess


import numpy as np


import pytest


from PIL import Image


from benchmarks.ocr_metrics import distance


from realopen_lens.inference import image_rgb


from realopen_lens.ocr import extract, parse_tsv


from realopen_lens.retrieval import (
    hybrid_rank,
    lexical_evidence,
    lexical_index,
    rank_lexical,
    vector_scores,
)


@pytest.mark.parametrize("mode", ["RGBA", "P"])
def test_transparent_screenshot_border_is_white_not_black(tmp_path, mode):
    # Rich's empty-OCR failure was caused by discarding transparent black pixels.
    path = tmp_path / "terminal.png"
    source = Image.new("RGBA", (12, 10), (0, 0, 0, 0))
    source.paste((35, 35, 35, 255), (3, 3, 9, 7))
    if mode == "P":
        source = source.quantize()
    source.save(path)
    original = path.read_bytes()
    with image_rgb(path) as actual:
        assert actual.mode == "RGB"
        assert actual.getpixel((0, 0)) == (255, 255, 255)
        assert actual.getpixel((5, 5)) == (35, 35, 35)
        assert not actual.info
    assert path.read_bytes() == original


def test_code_identifiers_are_preserved_and_not_split_into_generic_words():
    database = lexical_index(
        [
            {"image_id": "exact", "text": "divide_all connection_error"},
            {"image_id": "generic", "text": "divide all connection error"},
        ]
    )
    try:
        assert rank_lexical(database, "divide_all") == ["exact"]
        assert rank_lexical(database, "connection_error") == ["exact"]
        assert lexical_evidence("divide_all", "we divide all numbers") == 0
        assert lexical_evidence("divide_all", "call divide_all now") > 1
    finally:
        database.close()


def test_weak_ocr_does_not_bury_the_best_visual_result():
    # Previously, ANY two rank-50 votes beat one rank-1 visual vote.
    ids = ["photo"] + [f"ocr-{i}" for i in range(60)]
    visual = np.array([[0.2]] + [[0.1]] * 60, dtype=np.float32)
    passages = np.full((60, 1), 0.79, dtype=np.float32)
    ranking = hybrid_rank(
        visual,
        np.array([1]),
        ids,
        passages,
        np.array([1]),
        ids[1:],
        "a dog playing on the beach",
        [{"image_id": key, "text": "server logs connection failed"} for key in ids[1:]],
    )
    assert ranking[0] == "photo"


def test_correlated_ocr_channels_get_one_bounded_contribution():
    ranking = hybrid_rank(
        np.array([[0.18], [0.01]]),
        np.array([1]),
        ["photo", "text"],
        np.array([[0.99], [0.7]]),
        np.array([1]),
        ["text", "unrelated"],
        "database connection",
        [{"image_id": "text", "text": "database connection"}],
    )
    # One max(lexical, semantic) boost .16 leaves .18 visual ahead of .01+.16.
    assert ranking == ["photo", "text"]


def test_retrieval_rejects_misaligned_and_nonfinite_vectors():
    with pytest.raises(ValueError, match="identifiers"):
        vector_scores(np.ones((2, 1)), np.ones(1), ["only-one"])
    with pytest.raises(ValueError, match="finite"):
        vector_scores(np.array([[np.nan]]), np.ones(1), ["a"])


def test_tsv_keeps_reading_order_confidence_and_word_boxes():
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t70\t15\t90.5\tnotify_hello\n"
        "5\t1\t1\t1\t2\t1\t10\t40\t60\t15\t81.5\tDeadpool\n"
    )
    result = parse_tsv(tsv)
    assert result["text"] == "notify_hello\nDeadpool"
    assert result["words"][0]["box"] == [10, 20, 70, 15]
    assert result["confidence"] == 86


def test_failed_ocr_fallback_keeps_first_result_and_reports_warning(tmp_path, monkeypatch):
    path = tmp_path / "sparse.png"
    Image.new("RGB", (50, 50), "white").save(path)
    monkeypatch.setattr("realopen_lens.ocr.shutil.which", lambda _: "test-tesseract")
    monkeypatch.setattr("realopen_lens.ocr._environment", lambda *args: {"version": "test"})
    calls = []

    def recognize(*args):
        calls.append(args)
        if len(calls) > 1:
            raise subprocess.TimeoutExpired("test-tesseract", 1)
        return {
            "text": "one",
            "words": [{"text": "one", "confidence": 40, "box": [0, 0, 1, 1]}],
            "confidence": 40,
            "psm": 3,
            "seconds": 0,
            "stderr": "",
        }

    monkeypatch.setattr("realopen_lens.ocr._recognize", recognize)
    result = extract(path)
    assert result["text"] == "one"
    assert len(calls) == 2
    assert any("fallback failed" in warning for warning in result["warnings"])


def test_fast_edit_distance_matches_reference_on_random_unicode_sequences():
    def reference(left, right):
        previous = list(range(len(right) + 1))
        for i, a in enumerate(left, 1):
            row = [i]
            for j, b in enumerate(right, 1):
                row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
            previous = row
        return previous[-1]

    rng = random.Random(42)
    for _ in range(250):
        left = "".join(rng.choices("abéش_", k=rng.randrange(35)))
        right = "".join(rng.choices("abéش_", k=rng.randrange(35)))
        assert distance(left, right) == reference(left, right)
        assert distance(list(left), list(right)) == reference(left, right)


def test_diagram_layout_checks_detect_missing_lower_labels_without_numpy_scalars():
    import json
    from realopen_lens.ocr import _layout_incomplete, _remove_rules
    from PIL import ImageDraw

    image = Image.new("L", (300, 200), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 90, 290, 150), outline=0, width=2)
    draw.rectangle((30, 110, 70, 120), fill=0)
    incomplete = _layout_incomplete(image, [{"box": [10, 10, 100, 20]}])
    assert incomplete is True
    json.dumps({"incomplete": incomplete})
    cleaned, changed = _remove_rules(image)
    assert changed
    assert cleaned.getpixel((50, 90)) == 255
    assert cleaned.getpixel((50, 115)) == 0
    assert (
        _layout_incomplete(Image.new("L", (300, 200), 255), [{"box": [10, 10, 100, 20]}]) is False
    )


def test_tsv_rejects_nonfinite_confidence_instead_of_serializing_nan():
    header = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
    )
    with pytest.raises(ValueError, match="confidence"):
        parse_tsv(header + "5\t1\t1\t1\t1\t1\t1\t1\t20\t10\tnan\tword\n")
    with pytest.raises(ValueError, match="header"):
        parse_tsv("not tsv")


def test_relative_corpus_paths_and_invalid_limits(tmp_path, monkeypatch):
    from realopen_lens.corpus import scan

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "good.png").touch()
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    assert scan(Path("images"), 1) == ["good.png"]
    with pytest.raises(ValueError, match="positive"):
        scan(Path("images"), 0)


def test_literal_text_matching_handles_camelcase_normalization_and_boundaries():
    assert lexical_evidence("ModuleNotFoundError", "ModuleNotFoundError: psycopg") > 1
    assert lexical_evidence("ModuleNotFoundError", "OtherModuleNotFoundErrorExtra") == 0
    assert lexical_evidence("ｄｉｖｉｄｅ＿ａｌｌ", "call divide_all now") > 1
    assert lexical_evidence('"AI"', "AI tools") > 1
    assert lexical_evidence('"AI"', "chair") == 0
    assert lexical_evidence('"connection refused"', "connection\nrefused") > 1


@pytest.mark.parametrize("improves", [True, False])
def test_orientation_is_verified_and_boxes_describe_the_selected_input(
    tmp_path, monkeypatch, improves
):
    path = tmp_path / "rotated.png"
    Image.new("RGB", (100, 200), "white").save(path)
    monkeypatch.setattr("realopen_lens.ocr.shutil.which", lambda _: "test-tesseract")
    monkeypatch.setattr("realopen_lens.ocr._environment", lambda *args: {"version": "test"})
    monkeypatch.setattr(
        "realopen_lens.ocr._orientation", lambda *args: {"rotation": 90, "confidence": 8}
    )
    calls = []

    def recognize(image, *args):
        calls.append(image.size)
        confidence = 95 if len(calls) == 3 and improves else 20 if len(calls) == 3 else 30
        words = [
            {
                "text": "recovered" if confidence == 95 else "x",
                "confidence": confidence,
                "box": [1, 1, 5, 5],
            }
            for _ in range(8)
        ]
        return {
            "text": " ".join(w["text"] for w in words),
            "words": words,
            "confidence": confidence,
            "psm": args[-2],
            "ocr_size": list(image.size),
            "seconds": 0,
            "stderr": "",
        }

    monkeypatch.setattr("realopen_lens.ocr._recognize", recognize)
    result = extract(path)
    assert calls == [(100, 200), (100, 200), (200, 100)]
    assert result["rotation"] == (90 if improves else 0)
    assert result["ocr_size"] == ([200, 100] if improves else [100, 200])


def test_oversized_encoded_file_is_rejected_before_decode(tmp_path):
    path = tmp_path / "large.png"
    with path.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="encoded size"):
        image_rgb(path)


def test_confident_observed_alternatives_support_code_search_without_changing_transcript(
    tmp_path, monkeypatch
):
    path = tmp_path / "code.png"
    Image.new("RGB", (50, 50), "white").save(path)
    monkeypatch.setattr("realopen_lens.ocr.shutil.which", lambda _: "test-tesseract")
    monkeypatch.setattr("realopen_lens.ocr._environment", lambda *args: {"version": "test"})
    calls = []

    def recognize(image, *args):
        calls.append(1)
        words = (
            [{"text": "divide_al1", "confidence": 95, "box": [1, 1, 10, 10]}]
            if len(calls) == 1
            else [
                {"text": "divide_all", "confidence": 75, "box": [1, 1, 10, 10]},
                {"text": "weak_guess", "confidence": 10, "box": [1, 15, 10, 10]},
            ]
        )
        return {
            "text": " ".join(w["text"] for w in words),
            "words": words,
            "confidence": sum(w["confidence"] for w in words) / len(words),
            "psm": args[-2],
            "seconds": 0,
            "stderr": "",
        }

    monkeypatch.setattr("realopen_lens.ocr._recognize", recognize)
    result = extract(path)
    assert result["text"] == "divide_al1"
    assert result["alternate_texts"] == ["divide_all"]
    database = lexical_index([{"image_id": "code", **result}])
    try:
        assert rank_lexical(database, "divide_all") == ["code"]
        assert rank_lexical(database, "weak_guess") == []
    finally:
        database.close()


def test_common_form_vocabulary_cannot_outvote_a_photo_but_rare_terms_help():
    from realopen_lens.retrieval import lexical_frequencies

    records = [
        {"image_id": str(i), "text": "person above head address tax form"} for i in range(100)
    ]
    records.append({"image_id": "alarm", "text": "Set an alarm for 08:45"})
    database = lexical_index(records)
    try:
        query = "A person holding an umbrella above their head"
        evidence = lexical_evidence(query, records[0]["text"], lexical_frequencies(database, query))
        assert evidence <= 0.5
        query = "Set an alarm using a time input control"
        assert (
            lexical_evidence(query, records[-1]["text"], lexical_frequencies(database, query))
            >= 0.8
        )
        assert rank_lexical(database, "áddress")
    finally:
        database.close()


def test_unsupported_content_is_rejected_even_with_an_allowed_extension(tmp_path):
    path = tmp_path / "disguised.png"
    Image.new("RGB", (10, 10), "white").save(path, format="GIF")
    with pytest.raises(ValueError, match="Unsupported image content format"):
        image_rgb(path)
