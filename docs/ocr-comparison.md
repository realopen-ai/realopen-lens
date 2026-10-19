# Comparing OCR engines locally

Tesseract remains the default OCR backend. RapidOCR is an optional experiment under
`benchmarks/`; installing it does not switch application behavior or add it to the
production wheel dependencies. The experiment uses RapidOCR 3.10.0 with the bundled
PP-OCRv6 small detector/recognizer and legacy line orientation classifier.

Install the additional benchmark dependencies explicitly:

```sh
uv sync --locked --group ocr-eval --no-editable
```

The pinned RapidOCR wheel contains its three ONNX models. The adapter verifies
all three against `benchmarks/sources/rapidocr.lock.json` and supplies explicit
local paths. Neither inference worker downloads weights. Both engines use the
existing bounded image decoder. RapidOCR uses CPU execution, two threads, a
2400px longest edge and otherwise the pinned package defaults. Line polygons
remain line polygons; they are not mislabeled as Tesseract word boxes. Confidence
values between engines are not calibrated probabilities and should not be
compared as accuracy measures.

With the existing frozen corpus:

```sh
uv run --no-sync python -m benchmarks.ocr_compare --engines tesseract rapidocr --dataset data/benchmark-500-v2 --output runs/ocr-comparison/ppocrv6-small-v2 --threads 2
```

On macOS the experiment can be enclosed in an OS network-deny sandbox:

```sh
PYTHONPATH=src sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python -m benchmarks.ocr_compare --engines tesseract rapidocr --dataset data/benchmark-500-v2 --output runs/ocr-comparison/ppocrv6-small-v2 --threads 2
```

The same command resumes completed items. Use a new output directory if code,
models, dependencies, inputs or configuration change. No images are uploaded.
The parent supervises a single engine worker at a time. The current runner uses a
120s item deadline, 180s startup deadline and 14400s stage deadline; historical
two-engine runs used 90s, 120s and 3600s respectively. Native worker crashes/timeouts are
recorded individually and the worker restarts. Ordinary per-image errors do not
stop subsequent images. Recorded failures remain in the result set on resume.

Outputs:

- `config.json`: input metadata/hashes, corrected references, source fingerprints,
  model hashes, dependency versions, environment and resource settings.
- `results.sqlite`: durable results, retry history, current-item heartbeat,
  initialization environment and sampled memory for each engine.
- `tesseract.jsonl`, `rapidocr.jsonl`: every image, expected and actual text,
  available quality metrics, latency, RSS, errors/warnings and responsible stage.
- `changes.jsonl`: every image's before/after quality verdict and metrics.
- `summary.json`: groups by category, subcategory, format, resolution, difficulty,
  split and source kind. Logs and native recovery events remain alongside it.

The historical v2 annotation corrections are applied identically to both engines.
New v3 datasets already contain these corrections. The existing diagnostic gate
requires 80% on all available token recall, token precision and phrase recall
measures. CER/WER remain diagnostics because layout reading order is ambiguous.
Untranscribed images have no accuracy verdict; expected size-policy rejections
are reported separately from inferred images. Partially transcribed screenshots
have phrase scores only, not invented full-page CER/WER.

Both engines run fresh in the same environment. Tesseract's version and language
pack hashes are retained. RapidOCR's lazy models are initialized with a synthetic
warmup outside the image-quality and per-image-latency measurements. Initialization
and worker memory, including the warmup, are retained separately. Per-image RSS
includes resident models; sampling every 20ms can miss brief spikes and can count
shared pages twice. Empty OCR is not proof that an image contains no text.

This compares OCR pipelines only. It does not establish downstream retrieval
improvements, exhaustive multilingual accuracy, Windows/Linux performance, or
operation on an actual 8GB device. No script/language oracle is supplied to RapidOCR:
the same bundled recognition model handles every image. Its language routes include
English and French but **not Arabic**. The Arabic image is an explicit unsupported
script probe, not an Arabic accuracy test. Arabic requires a separate PP-OCRv5
recognizer or the existing Tesseract path; no such automatic routing is implemented.

Official references: [RapidOCR model list](https://rapidai.github.io/RapidOCRDocs/main/en/model_list/),
[local model configuration](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/how_to_use_offline_model/),
[model attribution](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/).

See the [completed 500-image comparison](benchmarks/2026-10-19-ocr-comparison.md)
for measured gains, regressions, memory costs and the Arabic support gap.

An explicitly selected [GLM-OCR backend through local Ollama](glm-ocr.md) is also
available. Its setup and three-engine comparison are separate from the CPU-only
experiment above.
