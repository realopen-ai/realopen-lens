"""Interrupted probes retain numerical outputs, not only progress messages."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.experiments import encoder_parity
from realopen_lens.encoders import onnx
from realopen_lens.models import manager


@pytest.fixture
def probe_setup(tmp_path, monkeypatch):
    folder = tmp_path / "models/e5/rev"
    folder.mkdir(parents=True)
    files = {}
    for key, name in (("text_int8", "model.onnx"), ("tokenizer", "tokenizer.json")):
        path = folder / name
        path.write_bytes(b"fixture")
        files[key] = {"path": name, "size": path.stat().st_size, "sha256": digest(path)}
    monkeypatch.setattr(
        manager, "registry", lambda: {"models": {"e5": {"revision": "rev", "files": files}}}
    )
    monkeypatch.setattr(encoder_parity, "enforce", lambda: None)
    control = {"interrupt": False, "metadata_failure": False}
    calls = []

    class Tokenizer:
        def encode(self, text):
            if control["metadata_failure"] and "second" in text:
                raise ValueError("Injected token metadata failure")
            return SimpleNamespace(ids=[1, 2])

    class Encoder:
        def __init__(self, *args):
            self.tokenizer = Tokenizer()

        def encode(self, texts, **kwargs):
            text = texts[0]
            calls.append(text)
            if control["interrupt"] and text == "second":
                raise KeyboardInterrupt
            result = np.zeros((1, 384), dtype=np.float32)
            result[0, ("first", "second", "third").index(text)] = 1
            return result

    monkeypatch.setattr(onnx, "TextEncoder", Encoder)
    inputs = tmp_path / "inputs.json"
    save_json(
        inputs,
        {
            "queries": [
                {"id": "سؤال-" + text, "text": text, "language": "ar"}
                for text in ("first", "second", "third")
            ]
        },
    )
    args = SimpleNamespace(
        inputs=inputs,
        models=tmp_path / "models",
        backend="int8",
        stage="e5-query",
        siglip_lowercase=False,
        threads=2,
        output=tmp_path / "probe",
    )
    return args, control, calls


def test_interrupted_probe_resumes_saved_vectors_without_reencoding_completed_inputs(probe_setup):
    args, control, calls = probe_setup
    control["interrupt"] = True
    with pytest.raises(KeyboardInterrupt):
        encoder_parity.worker(args)
    checkpoints = list((args.output / "items").glob("*.npz"))
    assert len(checkpoints) == 1 and not (args.output / "report.json").exists()
    saved = checkpoints[0].read_bytes()
    control["interrupt"] = False
    encoder_parity.worker(args)
    assert calls == ["first", "second", "second", "third"]
    assert checkpoints[0].read_bytes() == saved
    vectors = np.load(args.output / "vectors.npy", allow_pickle=False)
    assert np.array_equal(vectors[:, :3], np.eye(3))
    records = [
        json.loads(row)
        for row in (args.output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["vector_row"] for row in records] == [0, 1, 2]
    encoder_parity.worker(args)
    assert calls == ["first", "second", "second", "third"]


def test_metadata_failure_cannot_leave_a_stale_vector_row(probe_setup):
    args, control, calls = probe_setup
    control["metadata_failure"] = True
    with pytest.raises(ValueError, match="1 encoder inputs failed"):
        encoder_parity.worker(args)
    records = [
        json.loads(row)
        for row in (args.output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[1]["status"] == "error" and "vector_row" not in records[1]
    assert [row["vector_row"] for row in records if row["status"] == "ok"] == [0, 1]
    vectors = np.load(args.output / "vectors.npy", allow_pickle=False)
    assert vectors.shape == (2, 384) and np.array_equal(vectors[:, :3], np.eye(3)[[0, 2]])
