# Text-region detection with Tesseract

Text localization is an optional stage before Tesseract recognition. It uses the
pinned PP-OCRv6 small detector through CPU ONNX Runtime; it does not run RapidOCR's
recognizer. GLM and Ollama are not used in this evaluation or processing path.

The [completed Tesseract-only comparison](benchmarks/2026-10-19-tesseract-text-regions.md)
includes all 1608 case results, the fragment-merging fix, resource costs and
remaining masking regressions.

The existing whole-image Tesseract path remains the default when no detector is
selected. Enabling detection defaults to **gate** mode: an image without detected
regions skips recognition, with an explicit warning that text absence has not
been established. Otherwise, the existing OCR pipeline receives the original pixels.

**Mask** mode is experimental: it retains pixels inside expanded text regions and
replaces the surrounding background. It preserves the original coordinate system
and keeps nearby fragments of the same line together. It can remove real text or
introduce border artifacts, so it is not automatically selected over whole-image OCR.

## Setup and use

Install the optional image-processing dependency and explicitly fetch the detector:

```sh
uv sync --locked --extra text-regions --no-editable
uv run --no-sync lens models fetch-text-detector
```

The detector is about 10MB, downloaded only during explicit setup. Inference
requires the pinned local size and SHA-256; it never downloads missing weights or
silently substitutes another model. The detector runs on CPU with two inference
threads and a 1600px longest-edge bound. Recognition still uses the configured
Tesseract language packs (`eng+fra+ara` by default).

```sh
lens ocr data/example.png
lens ocr data/example.png --detector-model models/text-detector/PP-OCRv6_det_small.onnx
lens ocr data/example.png --detector-model models/text-detector/PP-OCRv6_det_small.onnx --region-mode mask
```

Results retain detector rectangles and scores, input dimensions, timing, source
coordinates and any skipped recognition. Scores describe localization evidence;
they are not calibrated probabilities of text absence or OCR correctness. Word
boxes retain the existing Tesseract coordinate/rotation metadata. Detector errors
are reported as errors. If its candidate bound is exceeded, the pipeline preserves
the full image rather than silently discarding excess regions.

## Reproduce the Tesseract-only comparison

Prepare the existing [500-image corpus](large-benchmark.md), then build the
supplementary text-presence controls. Their font is pinned, and every generated
or reused input retains its checksum and provenance:

```sh
uv run --no-sync python -m benchmarks.text_region_controls
uv run --no-sync python -m benchmarks.text_regions --controls data/text-region-controls-v1 --output runs/text-regions/full
```

This runs whole-image Tesseract, detection plus whole-image gating, and detection
plus masking sequentially. It uses no GLM or alternate OCR recognizer. On macOS,
the inference run can be enclosed in an OS network-deny sandbox:

```sh
PYTHONPATH=src:. sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python -m benchmarks.text_regions --controls data/text-region-controls-v1 --output runs/text-regions/full
```

`--variants`, `--threads`, `--model`, `--dataset`, `--controls`, `--limit` and
`--output` configure the run. The same command resumes completed records;
changes to inputs, model, code, packages or settings require a new output directory.
A worker supervisor bounds individual cases and records errors/timeouts without
losing subsequent cases. SQLite and per-variant JSONL retain every case, transcript,
reference, detector rectangle, warning, available accuracy metric, latency and
sampled process-tree RSS. `summary.json` retains category/subcategory, format,
resolution, difficulty and source aggregates. Source/dependency snapshots are saved.

Empty-reference controls are scored by whether OCR invents any text; no CER is
invented for a zero-length reference. Referenced positive images use the existing
fixed diagnostic gate. Untranscribed photos remain unscored for accuracy. A few
manual negative examples are diagnostic observations, not a representative estimate
of false-positive rates across all photos. The control set was used during
implementation and is not an untouched holdout set.

Detector provenance and model checksum are in
`src/realopen_lens/text-detector.lock.json`. Upstream references:
[model registry](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/python/rapidocr/default_models.yaml),
[DB preprocessing and postprocessing](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/python/rapidocr/ch_ppocr_det/utils.py).
