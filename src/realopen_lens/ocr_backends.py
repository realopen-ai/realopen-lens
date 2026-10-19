"""Explicit OCR engine selection; Tesseract is always the default."""

from pathlib import Path
from typing import Any


class TesseractBackend:
    def __init__(
        self, languages: str = "eng+fra+ara", threads: int = 2, detector=None, region_mode="gate"
    ):
        self.languages, self.threads = languages, threads
        self.detector, self.region_mode = detector, region_mode

    def extract(self, path: Path) -> dict:
        from realopen_lens.ocr import extract

        if self.detector is None:
            return extract(path, self.languages, self.threads)
        return extract(
            path,
            self.languages,
            self.threads,
            detector=self.detector,
            region_mode=self.region_mode,
        )

    def close(self) -> None:
        pass


def create_backend(
    engine: str = "tesseract",
    *,
    languages: str = "eng+fra+ara",
    threads: int = 2,
    detector_model: Path | None = None,
    region_mode: str = "gate",
    **glm_options: Any,
):
    if engine == "tesseract":
        detector = None
        if detector_model is not None:
            from realopen_lens.text_regions import TextRegionDetector

            detector = TextRegionDetector(detector_model, threads)
        return TesseractBackend(languages, threads, detector, region_mode)
    if detector_model is not None:
        raise ValueError("Text-region OCR is currently supported only with Tesseract")
    if engine == "glm-ocr":
        from realopen_lens.ollama_ocr import OllamaOCR

        return OllamaOCR(threads=threads, **glm_options)
    raise ValueError("Choose 'tesseract' or 'glm-ocr'; RapidOCR is benchmark-only experimental")
