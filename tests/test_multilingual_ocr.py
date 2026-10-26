"""New script support, legacy artifact compatibility and explicit OCR language costs."""

import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata

import pytest

from test_collections import app as _app

from realopen_lens.retrieval import ranking, text

from realopen_lens.ocr import runtime
from realopen_lens.runtime.readiness import suggested_languages
from benchmarks.experiments.multilingual_ocr import language_set, normalized

app = _app

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/sources/multilingual-ocr-v1"


@pytest.mark.parametrize(
    "case",
    json.loads((SOURCE / "lexical.json").read_text(encoding="utf-8")),
    ids=lambda case: case["id"],
)
def test_new_script_queries_rank_targets_above_distractors(case):
    index = ranking.lexical_index(
        [{"image_id": "a", "text": case["distractor"]}, {"image_id": "z", "text": case["text"]}]
    )
    try:
        found = ranking.rank_lexical(index, case["query"])
        assert found[0] == "z"
        if case["language"] in {"hin", "ben"}:
            assert found == ["z"]
    finally:
        index.close()


@pytest.mark.parametrize("value", ["किताब", "দিন", "청구서", "がっこう"])
def test_canonical_unicode_forms_match_without_losing_vowels(value):
    assert text.tokens(value) == text.tokens(unicodedata.normalize("NFD", value))
    assert text.normalize(value) == unicodedata.normalize("NFC", value)


def test_cjk_ocr_spacing_is_searchable_without_joining_korean_or_latin_words():
    assert text.normalize("连 接 失 败") == text.normalize("连接失败")
    assert text.normalize("한 글 text words") == "한 글 text words"
    index = ranking.lexical_index([{"image_id": "a", "text": "连 接 失 败"}])
    try:
        assert ranking.rank_lexical(index, "连接失败") == ["a"]
    finally:
        index.close()


def test_legacy_fts_is_rebuilt_in_memory_without_changing_saved_bytes(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    database = sqlite3.connect(path)
    database.execute("CREATE VIRTUAL TABLE ocr USING fts5(image_id UNINDEXED,text)")
    database.execute("INSERT INTO ocr VALUES ('a','网络连接失败请重试')")
    database.commit()
    database.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    current = ranking.open_lexical(path, [{"image_id": "a", "text": "网络连接失败请重试"}])
    try:
        assert ranking.rank_lexical(current, "连接失败") == ["a"]
        assert current.execute("PRAGMA user_version").fetchone()[0] == ranking.LEXICAL_VERSION
    finally:
        current.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_multilingual_query_limit_remains_bounded():
    index = ranking.lexical_index([])
    try:
        with pytest.raises(ValueError, match="256"):
            ranking.rank_lexical(index, "".join(chr(0x4E00 + i) for i in range(300)))
    finally:
        index.close()


def test_bundled_registry_and_corpus_cover_all_proposed_languages():
    registry = json.loads(
        (SOURCE.parents[2] / "packaging/tessdata.json").read_text(encoding="utf-8")
    )
    texts = json.loads((SOURCE / "texts.json").read_text(encoding="utf-8"))
    from realopen_lens.ocr.languages import BUNDLED_PACKS

    assert len(texts) == 19 and set(registry) == set(texts) | {"osd"} == BUNDLED_PACKS
    assert all(
        "/4.1.0/" in spec["url"] and len(spec["sha256"]) == 64 and spec["size"] > 0
        for spec in registry.values()
    )
    assert suggested_languages(list(registry)) == "eng+fra+ara"
    row = {"language": "hin", "script": "devanagari"}
    assert language_set("matched", row, list(texts)) == "hin+eng"
    assert language_set("script", row, list(texts)) == "eng+hin"
    assert len(language_set("all", row, list(texts)).split("+")) == 19
    assert normalized("किताब") != normalized("कतब")


def test_unused_bundled_language_packs_do_not_change_engine_identity(tmp_path, monkeypatch):
    engine = tmp_path / "bin/tesseract"
    resources = tmp_path / "resources"
    resources.mkdir()
    manifest = {
        "engine": "tesseract 5",
        "files": [{"file": "lens-ocr/bin/tesseract", "sha256": "engine"}],
        "tsv_sha256": "tsv",
        "tessdata": {"eng": {"sha256": "eng"}},
    }
    path = resources / "runtime.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runtime, "bundle_paths", lambda: (engine, resources))
    before = runtime.fingerprint(str(engine))
    manifest["tessdata"]["hin"] = {"sha256": "hin"}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert runtime.fingerprint(str(engine)) == before


def test_language_change_reuses_vision_and_metadata_stages(app):
    from test_collections import image

    root, store, calls, run = app
    image(root / "one.png")
    run()
    calls.clear()
    result = run(languages="hin+eng")
    assert [stage for stage, value in calls] == ["ocr", "semantic"]
    assert result.reused_stages == {"metadata": 1, "ocr": 0, "vision": 1, "semantic": 0}


def test_korean_retry_recovers_the_recorded_layout_failure(monkeypatch):
    import os
    from realopen_lens.ocr import tesseract
    from benchmarks.metrics.ocr import distance

    location = os.environ.get("LENS_MULTILINGUAL_TESSDATA")
    if not location:
        pytest.skip("Set LENS_MULTILINGUAL_TESSDATA to run the native pinned-pack regression")
    monkeypatch.setenv("TESSDATA_PREFIX", location)
    tesseract._environment.cache_clear()
    try:
        result = tesseract.extract(
            Path(__file__).parent / "fixtures/multilingual-ocr/korean-document.png", "kor+eng"
        )
        expected = normalized("청구서 번호 47291\n총 금액 125 유로\n오늘 결제를 받았습니다")
        assert distance(expected, normalized(result["text"])) / len(expected) < 0.15
        assert result["selected_psm"] == 6
        assert len(result["attempts"]) <= 4
    finally:
        tesseract._environment.cache_clear()
