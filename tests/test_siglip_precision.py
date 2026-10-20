"""Mixed precision setup and actual worker graph selection without model weights."""

import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from realopen_lens.models.manager import artifacts


def test_mixed_model_selection_keeps_e5_int8():
    selected = {(name, key) for name, key, _ in artifacts("int8", "fp32")}
    assert selected == {
        ("siglip2", "vision_fp32"),
        ("siglip2", "text_fp32"),
        ("siglip2", "tokenizer"),
        ("e5", "text_int8"),
        ("e5", "tokenizer"),
    }
    assert artifacts("int8") == artifacts("int8", "int8")
    with pytest.raises(ValueError, match="SigLIP2 precision"):
        artifacts("int8", "fp16")


@pytest.mark.parametrize(
    "kind,precision,expected",
    [
        ("siglip2", "fp32", "text_fp32"),
        ("siglip2", "int8", "text_int8"),
        ("e5", "fp32", "text_int8"),
    ],
)
def test_query_worker_selects_matching_graph(monkeypatch, capsys, kind, precision, expected):
    from realopen_lens.runtime import worker
    from realopen_lens.encoders import onnx
    from realopen_lens.models import manager

    selected = []

    class Encoder:
        def __init__(self, graph, tokenizer, model, threads):
            selected.append(graph.name)

        def encode(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(onnx, "TextEncoder", Encoder)
    monkeypatch.setattr(manager, "artifact_path", lambda root, name, key: Path(key))
    monkeypatch.setattr("realopen_lens.runtime.offline.enforce", lambda: None)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["worker", "--query"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "models": "models",
                    "kind": kind,
                    "siglip_precision": precision,
                    "query": "pizza",
                    "threads": 2,
                }
            )
        ),
    )
    worker.main()
    assert selected == [expected]
    assert json.loads(capsys.readouterr().out) == [1.0, 0.0]


@pytest.mark.parametrize("precision", ["int8", "fp32"])
def test_index_encoder_selects_matching_vision_graph(monkeypatch, precision):
    from realopen_lens.indexing import processing
    from realopen_lens.encoders import onnx

    selected = []
    monkeypatch.setattr(processing, "artifact_path", lambda root, name, key: Path(key))
    monkeypatch.setattr(onnx, "VisionEncoder", lambda graph, threads: selected.append(graph.name))
    processing.encoder("vision", {"models": "models", "threads": 2, "siglip_precision": precision})
    assert selected == ["vision_" + precision]


def test_fp32_verification_never_falls_back_to_int8(tmp_path, monkeypatch):
    from realopen_lens.models import manager

    manifest = {"models": {}}
    for name, keys in {
        "siglip2": ("vision_int8", "vision_fp32", "text_int8", "text_fp32", "tokenizer"),
        "e5": ("text_int8", "tokenizer"),
    }.items():
        folder = tmp_path / name / "fixture"
        folder.mkdir(parents=True)
        entries = {}
        for key in keys:
            path = folder / key
            path.write_bytes(b"pinned fixture")
            entries[key] = {
                "path": key,
                "size": path.stat().st_size,
                "sha256": manager.digest(path),
            }
        manifest["models"][name] = {"revision": "fixture", "files": entries}
    monkeypatch.setattr(manager, "registry", lambda: manifest)
    manager.verify(tmp_path, "int8", "fp32")  # No FP32 E5 graph is present or required.
    graph = tmp_path / "siglip2/fixture/text_fp32"
    graph.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity check failed"):
        manager.verify(tmp_path, "int8", "fp32")
    graph.unlink()
    with pytest.raises(ValueError, match="Missing model file"):
        manager.verify(tmp_path, "int8", "fp32")
    manager.verify(tmp_path, "int8")  # Available INT8 weights do not mask FP32 failure.
