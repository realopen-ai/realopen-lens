"""Regression tests for text-localization geometry, conservative masking and gating."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from realopen_lens.ocr.tesseract import extract
from realopen_lens.ocr.regions import TextRegionDetector, mask_regions, probability_regions


def test_detector_requires_verified_local_weights_before_loading_runtime(tmp_path):
    with pytest.raises(ValueError, match="Missing text detector"):
        TextRegionDetector(tmp_path / "absent.onnx")
    model = tmp_path / "wrong.onnx"
    model.write_bytes(b"incorrect")
    with pytest.raises(ValueError, match="integrity"):
        TextRegionDetector(model)


def test_probability_regions_restore_source_coordinates_and_preserve_boundaries():
    cv2 = pytest.importorskip("cv2")

    probability = np.zeros((64, 128), dtype=np.float32)
    probability[10:16, 20:80] = 0.9
    regions, overflow = probability_regions(probability, (256, 128), cv2)
    assert not overflow and len(regions) == 1
    left, top, right, bottom = regions[0]["box"]
    assert 0 <= left < 40 and 0 <= top < 20
    assert 160 < right <= 256 and 32 < bottom <= 128
    assert regions[0]["score"] == pytest.approx(0.9)


def test_candidate_overflow_does_not_silently_drop_regions():
    fake = SimpleNamespace(
        findContours=lambda *args: ([None] * 2001, None),
        RETR_EXTERNAL=0,
        CHAIN_APPROX_SIMPLE=0,
    )
    regions, overflow = probability_regions(np.zeros((32, 32)), (32, 32), fake)
    assert regions == [] and overflow
    with Image.new("RGB", (40, 30), "black") as image:
        image.putpixel((0, 0), (255, 255, 255))
        with mask_regions(image, {"regions": [], "candidate_limit_reached": True}) as masked:
            assert masked.tobytes() == image.tobytes()


def test_mask_keeps_all_original_pixels_inside_source_rectangle():
    with Image.new("RGB", (100, 60), "white") as image:
        ImageDraw.Draw(image).rectangle((10, 10, 30, 30), fill="black")
        image.putpixel((90, 50), (0, 0, 0))
        detection = {"regions": [{"box": [8, 8, 34, 34]}], "candidate_limit_reached": False}
        with mask_regions(image, detection) as masked:
            assert masked.size == image.size
            assert masked.crop((8, 8, 34, 34)).tobytes() == image.crop((8, 8, 34, 34)).tobytes()
            assert masked.getpixel((90, 50)) == (255, 255, 255)


def test_no_detected_regions_skips_tesseract_without_claiming_proven_absence(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 60), "white").save(image)
    monkeypatch.setattr("realopen_lens.ocr.tesseract.shutil.which", lambda name: "tesseract")
    monkeypatch.setattr("realopen_lens.ocr.tesseract._environment", lambda *args: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("Recognition must not run on gated input")

    monkeypatch.setattr("realopen_lens.ocr.tesseract._recognize", forbidden)
    detector = SimpleNamespace(
        detect=lambda image: {"regions": [], "candidate_limit_reached": False}
    )
    result = extract(image, detector=detector)
    assert result["text"] == "" and result["words"] == []
    assert result["inference_skipped"] == "no_detected_text_regions"
    assert "absence is not established" in result["warnings"][-1]


def test_detector_failure_is_not_silently_converted_to_empty_success(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 60), "white").save(image)
    monkeypatch.setattr("realopen_lens.ocr.tesseract.shutil.which", lambda name: "tesseract")
    monkeypatch.setattr("realopen_lens.ocr.tesseract._environment", lambda *args: {})

    def broken(image):
        raise ValueError("Invalid detector probability map")

    with pytest.raises(ValueError, match="probability map"):
        extract(image, detector=SimpleNamespace(detect=broken))


def test_fragmented_small_line_merges_without_joining_neighboring_rows_or_columns():
    from realopen_lens.ocr.regions import merge_line_regions

    regions = [
        {"box": [7, 9, 79, 29], "score": 0.84},
        {"box": [75, 15, 181, 24], "score": 0.75},
        {"box": [193, 16, 210, 24], "score": 0.55},
        {"box": [211, 12, 241, 26], "score": 0.82},
        {"box": [7, 45, 100, 60], "score": 0.8},
        {"box": [600, 10, 700, 30], "score": 0.9},
    ]
    merged = merge_line_regions(regions)
    assert len(merged) == 3
    assert merged[0]["box"] == [7, 9, 241, 29]
    assert merged[0]["members"] == 4
    assert merged[0]["score"] == 0.55


def test_cached_text_detector_setup_does_not_access_network(tmp_path, monkeypatch):
    import hashlib
    from realopen_lens.models.manager import fetch_text_detector

    payload = b"verified detector fixture"
    spec = {"name": "det.onnx", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr("realopen_lens.ocr.regions.detector_spec", lambda: spec)
    path = tmp_path / "text-detector" / "det.onnx"
    path.parent.mkdir()
    path.write_bytes(payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("Verified local model setup must stay offline")

    monkeypatch.setattr("realopen_lens.models.manager.urllib.request.urlopen", forbidden)
    assert fetch_text_detector(tmp_path) == path


@pytest.mark.parametrize("payload", [b"too long", b"bad!"])
def test_bad_detector_download_is_not_published(tmp_path, monkeypatch, payload):
    import hashlib
    import io
    from realopen_lens.models.manager import fetch_text_detector

    spec = {
        "name": "det.onnx",
        "size": 4,
        "sha256": hashlib.sha256(b"good").hexdigest(),
        "url": "https://example.invalid/det.onnx",
    }
    monkeypatch.setattr("realopen_lens.ocr.regions.detector_spec", lambda: spec)
    monkeypatch.setattr(
        "realopen_lens.models.manager.urllib.request.urlopen", lambda *a, **k: io.BytesIO(payload)
    )
    with pytest.raises(ValueError):
        fetch_text_detector(tmp_path)
    assert not (tmp_path / "text-detector" / "det.onnx").exists()
    assert not (tmp_path / "text-detector" / "det.onnx.part").exists()


@pytest.mark.parametrize("mode", ["gate", "mask"])
def test_no_regions_after_downscaling_preserves_source_ocr_and_reload(tmp_path, monkeypatch, mode):
    from realopen_lens.ocr.tesseract import _ocr_source

    path = tmp_path / "tiny-text.png"
    with Image.new("RGB", (2400, 1350), "white") as image:
        ImageDraw.Draw(image).rectangle((45, 105, 220, 130), fill="black")
        image.save(path)
    monkeypatch.setattr("realopen_lens.ocr.tesseract.shutil.which", lambda name: "tesseract")
    monkeypatch.setattr("realopen_lens.ocr.tesseract._environment", lambda *args: {})
    monkeypatch.setattr("realopen_lens.ocr.tesseract._layout_incomplete", lambda *args: False)
    detector = SimpleNamespace(
        detect=lambda image: {
            "regions": [],
            "candidate_limit_reached": False,
            "input_size": [1600, 896],
        }
    )
    calls = []

    def recognize(image, executable, languages, threads, psm, timeout):
        calls.append(image.getpixel((50, 110)))
        return {
            "text": "LENS-7303 status timeout",
            "words": [
                {"text": text, "confidence": 99, "box": [45, 105, 50, 25]}
                for text in ["LENS-7303", "status", "timeout"]
            ],
            "confidence": 99,
            "psm": psm,
        }

    monkeypatch.setattr("realopen_lens.ocr.tesseract._recognize", recognize)
    result = extract(path, detector=detector, region_mode=mode)
    assert result["text"] == "LENS-7303 status timeout"
    assert calls == [(0, 0, 0)]
    assert "inference_skipped" not in result
    assert result["text_detection"]["whole_image_fallback"] == "no_regions_after_downscaling"
    assert any("downscaling" in warning for warning in result["warnings"])
    with _ocr_source(path, result["text_detection"], mode) as image:
        assert image.getpixel((50, 110)) == (0, 0, 0)
