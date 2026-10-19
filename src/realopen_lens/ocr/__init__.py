"""Local OCR engines and explicitly selected experimental backends."""

from .tesseract import extract as extract
from .tesseract import parse_tsv as parse_tsv

__all__ = ["extract", "parse_tsv"]
