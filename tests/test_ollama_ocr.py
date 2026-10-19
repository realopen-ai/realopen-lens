"""Local-only OCR routing and bounded Ollama response contracts, without model downloads."""

import json
import subprocess
import sys

import pytest

from realopen_lens.ocr_backends import TesseractBackend, create_backend
from realopen_lens.ollama_ocr import OllamaOCR, parse_response


def test_tesseract_is_default_and_rapidocr_remains_experimental(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default OCR must not contact Ollama")

    monkeypatch.setattr(OllamaOCR, "__init__", forbidden)
    assert isinstance(create_backend(), TesseractBackend)
    with pytest.raises(ValueError, match="experimental"):
        create_backend("rapidocr")


def test_raw_generated_text_is_retained_and_truncation_is_explicit():
    raw = "<table><tr><td>example</td></tr></table>"
    result = parse_response(
        {
            "done": True,
            "done_reason": "length",
            "response": raw,
            "eval_count": 50,
            "eval_duration": 1000,
        }
    )
    assert result["text"] == raw
    assert result["truncated"] and result["warnings"]
    assert result["confidence"] is None and result["words"] is None
    assert result["ollama_metrics"]["eval_duration"] == 1000
    for response in (
        {"done": False, "response": "x"},
        {"done": True, "response": None},
        {"done": True, "done_reason": "other", "response": "x"},
    ):
        with pytest.raises(ValueError):
            parse_response(response)


def test_local_model_inspection_rejects_missing_cloud_and_changed_models(monkeypatch):
    shown = {"capabilities": ["vision"], "details": {}}
    tags = [{"name": "glm-ocr:q8_0", "digest": "a" * 64, "size": 100}]

    def request(self, path, payload=None):
        return {
            "/api/version": {"version": "test"},
            "/api/tags": {"models": tags},
            "/api/show": shown,
        }[path]

    monkeypatch.setattr(OllamaOCR, "request", request)
    engine = OllamaOCR()
    assert engine.info["digest"] == "a" * 64
    with pytest.raises(ValueError, match="digest changed"):
        OllamaOCR(expected_digest="b" * 64)
    shown["remote_host"] = "https://example.invalid"
    with pytest.raises(ValueError, match="cloud"):
        OllamaOCR()
    shown.pop("remote_host")
    tags.clear()
    with pytest.raises(ValueError, match="explicitly run"):
        OllamaOCR()


def test_http_uses_only_numeric_loopback_and_does_not_follow_redirects(monkeypatch):
    calls = []

    class Response:
        status = 302

        def read(self, size):
            return b"redirect"

    class Connection:
        def __init__(self, host, port, timeout):
            calls.append((host, port, timeout))

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            calls.append("closed")

    monkeypatch.setattr("realopen_lens.ollama_ocr.http.client.HTTPConnection", Connection)
    engine = OllamaOCR.__new__(OllamaOCR)
    engine.port, engine.timeout = 11434, 90
    with pytest.raises(ValueError, match="HTTP 302"):
        engine.request("/api/version")
    assert calls == [("127.0.0.1", 11434, 90), "closed"]


def test_loopback_exception_keeps_external_and_other_local_ports_blocked():
    code = """
import sys,json
from realopen_lens.offline import enforce
enforce(loopback_port=11434)
sys.audit("socket.getaddrinfo", "127.0.0.1", 11434, 0, 0, 0)
sys.audit("socket.connect", None, ("127.0.0.1", 11434))
blocked=0
for event,args in [("socket.getaddrinfo",("example.com",443,0,0,0)),
                   ("socket.connect",(None,("127.0.0.1",11435))),
                   ("socket.connect",(None,("192.0.2.1",11434))),
                   ("socket.sendto",(None,b"x",("127.0.0.1",11434)))]:
    try:sys.audit(event,*args)
    except PermissionError:blocked+=1
print(json.dumps(blocked))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == 4


def test_external_server_memory_counts_each_process_once(monkeypatch):
    from types import SimpleNamespace
    from benchmarks.resources import MemoryMonitor

    class Process:
        def __init__(self, pid=None):
            self.pid = pid or 1

        def children(self, recursive=True):
            return [Process(2), Process(3)] if self.pid == 1 else [Process(3)]

        def memory_info(self):
            return SimpleNamespace(rss=self.pid * 100)

    monkeypatch.setattr("benchmarks.resources.psutil.Process", Process)
    monitor = MemoryMonitor(external_pids=(2,))
    monitor._sample()
    assert monitor.peaks["startup"] == 600
    assert monitor.samples == 1


def test_installed_cli_exposes_explicit_secondary_engine_and_defaults_to_tesseract(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from typer.testing import CliRunner
    from realopen_lens.cli import app

    calls = []
    image = tmp_path / "image.png"
    image.touch()

    def factory(engine, **kwargs):
        calls.append((engine, kwargs))
        return SimpleNamespace(extract=lambda path: {"text": "example"}, close=lambda: None)

    monkeypatch.setattr("realopen_lens.ocr_backends.create_backend", factory)
    runner = CliRunner()
    assert runner.invoke(app, ["ocr", str(image)]).exit_code == 0
    assert calls[-1][0] == "tesseract"
    result = runner.invoke(
        app, ["ocr", str(image), "--engine", "glm-ocr", "--ollama-port", "11435"]
    )
    assert result.exit_code == 0
    assert calls[-1][0] == "glm-ocr" and calls[-1][1]["port"] == 11435


@pytest.mark.parametrize("color", ["white", "#808080", "black"])
def test_uniform_pixels_do_not_trigger_generative_ocr(tmp_path, color):
    from PIL import Image

    path = tmp_path / "uniform.png"
    Image.new("RGB", (600, 600), color).save(path)
    engine = object.__new__(OllamaOCR)
    engine.loaded = lambda: {}

    def forbidden(image):
        raise AssertionError("Exactly uniform input must not trigger model generation")

    engine.generate = forbidden
    result = engine.extract(path)
    assert result["text"] == "" and result["inference_skipped"] == "uniform_pixels"
    assert result["done_reason"] is None and result["ollama_metrics"] == {}


def test_nonuniform_detail_is_checked_before_downsampling(tmp_path):
    from PIL import Image

    path = tmp_path / "detail.png"
    with Image.new("RGB", (4800, 4800), "white") as image:
        image.putpixel((20, 20), (0, 0, 0))
        image.save(path)
    engine = object.__new__(OllamaOCR)
    engine.loaded = lambda: {}
    calls = []

    def generate(image):
        calls.append(image.size)
        return {"text": "detail", "warnings": []}

    engine.generate = generate
    result = engine.extract(path)
    assert calls == [(2400, 2400)] and result["text"] == "detail"
    assert "inference_skipped" not in result
