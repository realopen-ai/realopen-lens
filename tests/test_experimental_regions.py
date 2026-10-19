"""Text localization requires explicit experimental selection; the default stays whole-image."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from realopen_lens.cli import app
from realopen_lens.ocr_backends import create_backend


def test_default_ocr_never_initializes_a_detector_or_marks_output_experimental(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default OCR must not load a detector")

    monkeypatch.setattr("realopen_lens.text_regions.TextRegionDetector", forbidden)
    calls = []
    monkeypatch.setattr(
        "realopen_lens.ocr.extract", lambda *args, **kw: calls.append(kw) or {"text": "whole"}
    )
    result = create_backend().extract(Path("input.png"))
    assert result == {"text": "whole"}
    assert calls == [{}]


@pytest.mark.parametrize("mode", ["gate", "mask"])
def test_selected_experiment_retains_output_and_marks_its_mode(monkeypatch, mode):
    detector = object()
    monkeypatch.setattr("realopen_lens.text_regions.TextRegionDetector", lambda *args: detector)
    calls = []
    monkeypatch.setattr(
        "realopen_lens.ocr.extract",
        lambda *args, **kw: calls.append(kw) or {"text": "readable", "warnings": ["diagnostic"]},
    )
    result = create_backend(detector_model=Path("local.onnx"), region_mode=mode).extract(
        Path("input.png")
    )
    assert result["text"] == "readable" and result["warnings"] == ["diagnostic"]
    assert result["experimental"] == {"text_regions": mode}
    assert calls == [{"detector": detector, "region_mode": mode}]


@pytest.mark.parametrize(
    "options, message",
    [
        ({"region_mode": "mask"}, "requires --experimental"),
        ({"region_mode": "typo", "detector_model": Path("local.onnx")}, "must be"),
    ],
)
def test_invalid_experiment_options_fail_before_loading_models(monkeypatch, options, message):
    def forbidden(*args):
        raise AssertionError("Invalid options must not load models")

    monkeypatch.setattr("realopen_lens.text_regions.TextRegionDetector", forbidden)
    with pytest.raises(ValueError, match=message):
        create_backend(**options)


def test_cli_experiment_is_explicit_and_does_not_contaminate_json(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    image.touch()
    calls = []

    def factory(engine, **options):
        calls.append(options)
        output = {"text": "example"}
        if options["detector_model"] is not None:
            output["experimental"] = {"text_regions": options["region_mode"]}
        return SimpleNamespace(extract=lambda path: output, close=lambda: None)

    monkeypatch.setattr("realopen_lens.ocr_backends.create_backend", factory)
    runner = CliRunner()
    default = runner.invoke(app, ["ocr", str(image)])
    assert default.exit_code == 0 and json.loads(default.stdout) == {"text": "example"}
    assert not default.stderr and calls[-1]["detector_model"] is None
    selected = runner.invoke(
        app, ["ocr", str(image), "--experimental-detector-model", "local.onnx"]
    )
    assert selected.exit_code == 0
    assert "Experimental text-region OCR" in selected.stderr
    assert json.loads(selected.stdout)["experimental"] == {"text_regions": "gate"}
    assert calls[-1]["detector_model"] == Path("local.onnx")
    assert runner.invoke(app, ["ocr", str(image), "--detector-model", "local.onnx"]).exit_code == 2
