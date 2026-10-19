"""Compatibility worker entry point, also loaded by preserved-baseline evaluation."""

from benchmarks.runtime.baseline_worker import ocr as ocr
from benchmarks.runtime.baseline_worker import main

__all__ = ["ocr", "main"]

if __name__ == "__main__":
    main()
