"""Bounded image-query protocol and public CLI failure handling."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from realopen_lens.cli import app
from realopen_lens.retrieval import images


@pytest.mark.parametrize("value", [[0] * 768, [1] * 384, [float("nan")] * 768])
def test_image_worker_rejects_invalid_embedding(monkeypatch, value):
    monkeypatch.setattr(
        images.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(value))
    )
    with pytest.raises(ValueError, match="Invalid image query embedding"):
        images.image_vector(Path("query.png"), Path("models"), 2, "int8")


def test_image_worker_protocol_is_bounded_and_selects_precision(monkeypatch, tmp_path):
    def run(command, **kwargs):
        assert command[-1] == "--image-query"
        assert kwargs["timeout"] == 120 and kwargs["check"]
        request = json.loads(kwargs["input"])
        assert request["siglip_precision"] == "fp32" and request["threads"] == 2
        assert request["models"] == str(tmp_path)
        return SimpleNamespace(stdout=json.dumps([2] + [0] * 767))

    monkeypatch.setattr(images.subprocess, "run", run)
    vector = images.image_vector(Path("query.png"), tmp_path, 2, "fp32")
    np.testing.assert_array_equal(vector, [1] + [0] * 767)


@pytest.mark.parametrize("timeout", [True, False])
def test_image_worker_reports_timeout_and_crash(monkeypatch, timeout):
    def run(*args, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired("worker", 120)
        raise subprocess.CalledProcessError(1, "worker", stderr="decode failure")

    monkeypatch.setattr(images.subprocess, "run", run)
    with pytest.raises(TimeoutError if timeout else RuntimeError):
        images.image_vector(Path("query.png"), Path("models"), 2, "int8")


@pytest.mark.parametrize("args", [[], ["no-file.png", "--image-id", "abc"], ["no-file.png"]])
def test_similar_cli_returns_clean_errors(args):
    result = CliRunner().invoke(app, ["similar", *args])
    assert result.exit_code == 1
    assert "Error:" in result.output and "Traceback" not in result.output


def test_external_image_compatibility_checks_pillow_and_only_vision_graph(monkeypatch, tmp_path):
    from importlib.metadata import version
    from realopen_lens.encoders import onnx
    from realopen_lens.models.manager import digest, registry

    models = registry()
    graph = tmp_path / "vision.onnx"
    graph.write_bytes(b"graph")
    monkeypatch.setattr(images, "artifact_path", lambda *args: graph)
    monkeypatch.setattr(images, "registry", lambda: models)
    models["models"]["siglip2"]["files"]["vision_fp32"] = {
        "size": graph.stat().st_size,
        "sha256": digest(graph),
    }
    spec = {
        "model_registry": models,
        "inference_sha256": digest(Path(onnx.__file__)),
        "packages": {name: version(name) for name in ("numpy", "Pillow", "onnxruntime")},
    }
    images.verify_image_encoder(spec, tmp_path, "fp32")
    spec["packages"]["Pillow"] = "different"
    with pytest.raises(ValueError, match="dependencies changed"):
        images.verify_image_encoder(spec, tmp_path, "fp32")
    spec["packages"]["Pillow"] = version("Pillow")
    graph.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="integrity"):
        images.verify_image_encoder(spec, tmp_path, "fp32")
    graph.unlink()
    with pytest.raises(ValueError, match="Missing model"):
        images.verify_image_encoder(spec, tmp_path, "fp32")
    spec["inference_sha256"] = "different"
    with pytest.raises(ValueError, match="preprocessing version differs"):
        images.verify_image_encoder(spec, tmp_path, "fp32")


def test_image_query_worker_uses_vision_graph_and_shared_preprocessing(
    monkeypatch, capsys, tmp_path
):
    import io
    import sys
    from realopen_lens.runtime import worker
    from realopen_lens.encoders import onnx
    from realopen_lens.models import manager

    graph = tmp_path / "fp32.onnx"
    image = tmp_path / "query.png"
    monkeypatch.setattr(
        manager,
        "artifact_path",
        lambda root, kind, key: (
            graph if kind == "siglip2" and key == "vision_fp32" else pytest.fail("wrong graph")
        ),
    )

    class Encoder:
        def __init__(self, path, threads):
            assert path == graph and threads == 2

        def encode(self, paths):
            assert paths == [image]
            return np.array([[1] + [0] * 767], dtype=np.float32)

    monkeypatch.setattr(onnx, "VisionEncoder", Encoder)
    monkeypatch.setattr("realopen_lens.runtime.offline.enforce", lambda: None)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["worker", "--image-query"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "image": str(image),
                    "models": str(tmp_path),
                    "threads": 2,
                    "siglip_precision": "fp32",
                }
            )
        ),
    )
    worker.main()
    assert json.loads(capsys.readouterr().out) == [1] + [0] * 767


def test_image_query_rejects_oversize_before_reading_or_opening_index(tmp_path):
    path = tmp_path / "huge.png"
    with path.open("wb") as f:
        f.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="64 MiB"):
        images.search_similar(path, index_dir=tmp_path / "absent-index")


@pytest.mark.parametrize("use_id", [False, True])
def test_similar_cli_forwards_options_and_serializes_results(monkeypatch, tmp_path, use_id):
    from realopen_lens import search
    from realopen_lens.search import SearchResult

    query = tmp_path / "query.png"

    def similar(image=None, **options):
        assert image == (None if use_id else query)
        assert options["image_id"] == ("abc" if use_id else None)
        assert options["include_self"] and options["limit"] == 3
        assert options["siglip_precision"] == "fp32"
        assert options["collection"] == "photos"
        return [SearchResult(1, "abc", ("photo.png",), None, 20, 30, "", 0.8)]

    monkeypatch.setattr(search, "search_similar", similar)
    args = ["--image-id", "abc"] if use_id else [str(query)]
    result = CliRunner().invoke(
        app,
        [
            "similar",
            *args,
            "--include-self",
            "--limit",
            "3",
            "--siglip-precision",
            "fp32",
            "--collection",
            "photos",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["visual_similarity"] == 0.8


def test_image_worker_decode_failure_is_clean(monkeypatch, capsys):
    import io
    import sys
    from realopen_lens.runtime import worker
    from realopen_lens.encoders import onnx

    class Broken:
        def __init__(self, *args):
            pass

        def encode(self, *args):
            raise ValueError("unsupported image")

    monkeypatch.setattr(onnx, "VisionEncoder", Broken)
    monkeypatch.setattr("realopen_lens.runtime.offline.enforce", lambda: None)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["worker", "--image-query"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "image": "broken.png",
                    "models": "models",
                    "threads": 2,
                    "siglip_precision": "int8",
                }
            )
        ),
    )
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert not output.out and "unsupported image" in output.err and "Traceback" not in output.err
