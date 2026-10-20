"""Keep diagnostic expansion, normalization, and encoder comparisons auditable."""

from copy import deepcopy
import io
import json
import sys
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.experiments.arabic_lexical import fold
from benchmarks.experiments.encoder_parity import compare
from benchmarks.experiments.query_expansion import expand
from benchmarks.experiments.precision_retrieval import vectors
from benchmarks.experiments.ocr_recovery import run as run_ocr_probe, worker as ocr_worker
from realopen_lens.retrieval.ranking import lexical_index, rank_lexical


def request(language, pool, ranking):
    return {
        "status": "ok",
        "query_id": "family-" + language,
        "seconds": 1,
        "query": {
            "id": "family-" + language,
            "family": "family",
            "language": language,
            "split": "train",
            "judgments": [{"image_sha256": "target", "grade": 2}],
        },
        "trace": {
            "candidates": [{"image_id": key, "hybrid_score": score} for key, score in pool],
            "rankings": {mode: ranking for mode in ("visual", "semantic", "lexical", "hybrid")},
        },
    }


def test_reviewed_expansion_recovers_pool_target_and_deduplicates_language_votes():
    original = request("ar", [("noise", 0.5)], ["noise"])
    english = request("en", [("target", 0.8), ("noise", 0.4)], ["target", "noise"])
    record = expand(original, english)
    assert record["variant_ids"] == ["family-ar", "family-en"]
    assert record["pool_size"] == 2 and record["new_pool_positives"] == ["target"]
    assert record["metrics"]["original"]["mrr_at_10"] == 0
    assert record["metrics"]["max_hybrid"]["mrr_at_10"] == 1
    english_control = expand(english, english)
    assert english_control["variant_ids"] == ["family-en"]
    assert english_control["rankings"]["max_hybrid"] == english["trace"]["rankings"]["hybrid"]
    assert english_control["new_pool_positives"] == []
    wrong = deepcopy(english)
    wrong["query"]["split"] = "test"
    with pytest.raises(ValueError, match="same family/fold"):
        expand(original, wrong)


def test_arabic_character_folding_adds_a_view_without_changing_original_identifiers():
    text = "مكتـبة ERR_٤٢ کتب على"
    original = lexical_index([{"image_id": "target", "text": text}])
    normalized = lexical_index([{"image_id": "target", "text": fold(text)}])
    try:
        assert rank_lexical(original, "ERR_42") == []
        assert rank_lexical(normalized, fold("ERR_42")) == ["target"]
        assert rank_lexical(original, "مكتبة") == []
        assert rank_lexical(normalized, fold("مكتبة")) == ["target"]
        assert rank_lexical(original, "ERR_٤٢") == ["target"]
    finally:
        original.close()
        normalized.close()
    assert fold("/Users/Jean/ERR_42.py") == "/Users/Jean/ERR_42.py"


def probe(path, vector, ids):
    path.mkdir()
    np.save(path / "vectors.npy", np.asarray([vector], dtype=np.float32))
    row = {
        "id": "q-ar",
        "status": "ok",
        "vector_row": 0,
        "input_ids": ids,
        "language": "ar",
        "category": None,
    }
    (path / "records.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    save_json(
        path / "report.json",
        {
            "identity": {"inputs_sha256": "fixed", "stage": "siglip2"},
            "errors": 0,
            "inputs": 1,
            "vectors_sha256": digest(path / "vectors.npy"),
            "records_sha256": digest(path / "records.jsonl"),
        },
    )


def test_encoder_comparison_distinguishes_token_mismatch_and_rejects_changed_outputs(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    probe(left, [1, 0], [1, 2])
    probe(right, [0.8, 0.6], [1, 3])
    report = compare(left, right, tmp_path / "comparison")
    assert report["groups"]["all"]["cosine"]["mean"] == pytest.approx(0.8)
    assert report["groups"]["language:ar"]["token_mismatches"] == 1
    np.save(right / "vectors.npy", np.array([[1, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="outputs changed"):
        compare(left, right, tmp_path / "bad")


def test_precision_ablation_refuses_misaligned_image_or_query_embeddings(tmp_path):
    path = tmp_path / "probe"
    probe(path, [1, 0], [1, 2])
    values, report = vectors(path, [{"id": "q-ar"}])
    assert values.shape == (1, 2)
    with pytest.raises(ValueError, match="identities differ"):
        vectors(path, [{"id": "different-image"}])
    with pytest.raises(ValueError, match="Incomplete"):
        vectors(path, [{"id": "q-ar"}, {"id": "extra"}])


def test_ocr_worker_transmits_arabic_over_a_windows_default_encoding_pipe(tmp_path, monkeypatch):
    from realopen_lens.ocr import tesseract
    from realopen_lens.runtime import offline

    path = tmp_path / "image.png"
    path.write_bytes(b"fixture")
    row = {"path": str(path), "sha256": digest(path)}
    monkeypatch.setattr(tesseract, "extract", lambda *args: {"text": "نص عربي"})
    monkeypatch.setattr(offline, "enforce", lambda: None)
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding="cp1252", write_through=True)
    with monkeypatch.context() as context:
        context.setattr(sys, "stdin", io.StringIO(json.dumps(row)))
        context.setattr(sys, "stdout", stream)
        ocr_worker(SimpleNamespace(languages="ara", threads=1, timeout=1))
        encoded = output.getvalue()
    assert json.loads(encoded.decode("ascii"))["actual"]["text"] == "نص عربي"


def test_ocr_resume_refuses_a_changed_native_binary_even_if_version_is_unchanged(
    tmp_path, monkeypatch
):
    from realopen_lens.ocr import tesseract

    executable = tmp_path / "tesseract"
    executable.write_bytes(b"native-v1")
    monkeypatch.setattr(
        "benchmarks.experiments.ocr_recovery.shutil.which", lambda *args: str(executable)
    )

    @lru_cache()
    def environment(*args):
        return {"version": "fixture", "language_pack_sha256": {"eng": "frozen"}}

    monkeypatch.setattr(tesseract, "_environment", environment)
    source = tmp_path / "engine/realopen_lens/ocr/tesseract.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixture\n", encoding="utf-8")
    inputs = tmp_path / "empty-fixture.json"
    save_json(inputs, {"images": []})
    args = SimpleNamespace(
        inputs=inputs,
        output=tmp_path / "run",
        engine_root=tmp_path / "engine",
        languages="eng",
        threads=1,
        timeout=30,
    )
    run_ocr_probe(args)
    executable.write_bytes(b"native-v2")
    with pytest.raises(ValueError, match="resume identity changed"):
        run_ocr_probe(args)
