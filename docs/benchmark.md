# Inference benchmark

For the larger, resumable per-image suite, see the
[500-image benchmark guide](large-benchmark.md).

This harness tests local CPU inference and retrieval independently of the
[persistent collection workflow](collections.md). It does not train a model or
establish production hardware support.

The harness lives in the repository's `benchmarks/` directory, alongside sample
downloaders, generated fixtures, reference transcripts, and evaluation metrics.
It imports the application engine from `src/realopen_lens/`; the engine never
imports benchmark tooling. These tools are excluded from the application wheel
and require a repository checkout with development dependencies installed.
Run the commands below from the repository root.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked --no-editable
uv run --no-sync lens --help
```

Install Tesseract 5 and the OCR language packs you intend to test. The benchmark
looks for `tesseract` on `PATH`; it does not install native software for you.

- macOS with Homebrew: `brew install tesseract tesseract-lang`.
- Debian/Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-ara`.
- Windows: follow the [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html),
  select the required language data, and add its installation directory to `PATH`.

Inspect available providers and OCR languages:

```sh
uv run --no-sync lens doctor
```

Download pinned INT8 models and tokenizers explicitly (approximately 548 MB):

```sh
uv run --no-sync lens models fetch
uv run --no-sync lens models verify
```

Among the `lens` commands, only `models fetch` accesses the network. Model files
go into ignored `models/`. The optional sample downloader below also uses the
network explicitly.
The bundled `realopen_lens/models/models.lock.json` pins repository revisions, sizes, and SHA-256
checksums. Existing matching files are reused; interrupted partial downloads
restart on the next fetch. Inference never downloads missing files.

For an offline machine, copy the complete `models/` directory from a prepared
machine and verify it locally. Python dependencies and Tesseract must also be
installed in advance.

The setup command installs a regular package so it also works in environments
that hide editable-install `.pth` files. After changing source files, rebuild it
explicitly so uv does not reuse a previously built wheel:

```sh
uv sync --locked --no-editable --reinstall-package realopen-lens
```

## Use your images

Put images in the repository's root `data/` directory. Subfolders are supported.
`data/`, `models/`, `.cache/`, and `runs/` are ignored by Git. Supported inputs are
JPEG, PNG, WebP, BMP, and TIFF; only the first frame/page is processed. Files
resolving outside the collection through symlinks are excluded. Originals are
never changed.

```sh
uv run --no-sync python -m benchmarks run --data data --languages eng+fra+ara
```

The default selects the first 100 supported paths in sorted order, uses one image
per vision batch, and limits inference to two CPU threads. Use `--limit`,
`--batch-size`, and `--threads` to change these settings. Choose OCR packs matching
your images; listing additional languages does not guarantee OCR accuracy.

Add query judgments in `data/queries.json` to measure retrieval:

```json
[
  {
    "id": "postgres-error-en",
    "text": "Docker cannot connect to PostgreSQL",
    "language": "en",
    "relevant": ["screenshots/postgres-error.png"]
  },
  {
    "id": "postgres-error-fr",
    "text": "Erreur de connexion à la base de données",
    "language": "fr",
    "relevant": ["screenshots/postgres-error.png"]
  }
]
```

Replace the example paths with actual images. IDs are paths relative to `data/`,
using `/` separators on every platform. All judged images must be included in
the selected subset. An empty `relevant` list produces rankings without quality
metrics for that query; it does not declare that no relevant images exist.

```sh
uv run --no-sync python -m benchmarks run --data data --queries data/queries.json --output runs/first
```

Each run requires a new output directory to avoid mixing results. Without a query
file, the benchmark measures OCR and embedding generation only. Invalid images,
missing OCR packs, worker failures, and mismatched model checksums stop the run
with an error rather than silently altering the evaluation collection. Images
larger than 24 million decoded pixels are rejected for this initial benchmark.
OCR starts with an EXIF-oriented, alpha-composited RGB copy fitting 2400 × 2400
pixels, with bounded grayscale/layout/orientation fallbacks and a 3200-pixel
fallback for large text-rich inputs. Encoded files over
64 MiB are rejected.

## Download random public photographs

An optional script downloads a reproducible random sample of COCO photographs
into `data/public-sample/`, together with source URLs, checksums, captions, and
caption-derived queries. It uses a fixed random seed, verifies the pinned
annotation archive, and reuses verified images on subsequent runs. Existing
unverified images and edited query files are not overwritten.

```sh
uv run --no-sync python -m benchmarks.samples --count 50 --seed 42
uv run --no-sync python -m benchmarks run --data data/public-sample --queries data/public-sample/queries.caption-derived.json --corpus-kind public-coco-caption-derived
```

The first command uses the network; the benchmark runs locally. Source annotations
come from the [Karpathy caption dataset release](https://cs.stanford.edu/people/karpathy/deepimagesent/),
and JPEG files come from COCO's public image bucket. The script selects from the
5,000-image COCO 2014 Karpathy test split. The annotation archive is about 37 MB;
only the selected images are downloaded. `--count` supports up to 500 images;
set the benchmark's `--limit` accordingly when selecting more than 100.

Use `--output data/another-sample --seed 123` for a different sample. A directory
cannot be reused with a different seed or count. Failed downloads are recorded;
rerun the same command to retry missing images. Image and annotation provenance
is saved in `download-manifest.json`, including a link to COCO's source terms.
Downloaded assets remain ignored rather than being distributed with this repo.

Caption-derived English queries supply one known positive per image. Other images
may also be relevant, so these are provisional retrieval checks, not exhaustive
human judgments. This sample contains photographs, not screenshots or diagrams,
and provides no OCR transcription ground truth. Evaluate multilingual search
with a separate reviewed query file; the downloader does not call a translation
service or claim to generate multilingual annotations.

The [initial 50-image results](benchmarks/2026-10-18-public-sample.md) document
observed retrieval quality, resource use, and an equal-weight fusion limitation.

## Add terminal, document, and diagram images

The sourced OCR diagnostic set adds three published terminal screenshots, two
scanned article pages plus a printed OCR test image, and three C4 architecture
diagrams. It includes reference transcripts and 27 queries in English, French,
and Arabic, authored after inspecting the images.

```sh
uv run --no-sync python -m benchmarks.ocr_samples --include-photos
uv run --no-sync python -m benchmarks run --data data --queries data/queries.mixed.json --ocr-truth data/ocr-truth.mixed.json --languages eng+fra+ara --corpus-kind public-mixed-photos-terminal-scans-diagrams
```

`--include-photos` expects the previously downloaded `data/public-sample/` to
exist. It prepares combined root-relative annotations for those photos and
`data/ocr-sample/`; existing assets are reused only when checksums match. Edited
annotation files are not overwritten. The optional downloader requires `curl`
on `PATH`; inference does not. Every image and upstream document transcript is
checksum-pinned. If an upstream image changes, the script stops rather than
silently changing the evaluation data.

Without `--include-photos`, benchmark the nine images independently:

```sh
uv run --no-sync python -m benchmarks.ocr_samples
uv run --no-sync python -m benchmarks run --data data/ocr-sample --queries data/ocr-sample/queries.json --ocr-truth data/ocr-sample/ocr-truth.json --corpus-kind public-text-heavy
```

All nine images contain English text. French and Arabic query translations test
retrieval, not French or Arabic OCR. Document references come from upstream
Tesseract test data; terminal and diagram transcriptions were checked visually.
Decorative borders and prompt icons are excluded from terminal reference text.
Diagrams use node text followed by relationship labels and footers, so CER/WER
can penalize a different, otherwise valid reading order. Token recovery provides
an additional order-independent diagnostic. Sources and attribution are in the
repository’s `benchmarks/ocr_assets/manifest.json` and
`benchmarks/ocr_assets/ATTRIBUTION.md`.

The [first mixed-corpus results](benchmarks/2026-10-18-text-heavy.md) report OCR
recovery and retrieval separately for photos, terminals, documents, and diagrams.

## Generated smoke fixtures

To exercise the pipeline before adding a real collection:

```sh
uv run --no-sync python -m benchmarks fixture
uv run --no-sync python -m benchmarks run --data .cache/smoke-images --queries .cache/smoke-images/queries.json --ocr-truth .cache/smoke-images/ocr-truth.json --corpus-kind generated-smoke --output runs/smoke
```

The fixture generator draws simple document-like images in English and French.
It adds Arabic only when Pillow has RAQM support for correct shaping. Pass
`--font /path/to/font.ttf` if the default font lookup fails; Arabic rendering
requires a font with Arabic glyphs. `fixture-info.json` records whether Arabic
was rendered. These fixtures contain no real photographs and provide no evidence
of real-world retrieval quality or broad language support.

## Reports and measurement limits

`report.json` records host RAM/OS/architecture, package versions, model identities,
input fingerprints, stage timings, sampled process-tree memory, and rankings.
`report.md` provides a readable summary. Supporting files retain OCR text,
embeddings, judgments, and worker logs. They are private local artifacts; review
them before sharing.

Stages run in separate subprocesses: OCR, visual embedding generation, SigLIP2
query encoding, and E5 passage/query encoding. Their runtimes do not remain loaded
together. The coordinator samples its own RSS plus recursive children every
20 ms, including Tesseract processes. Shared memory may be counted twice, and
short spikes may be missed. Access failures are recorded, not interpreted as zero
memory. The measured machine's RAM is reported; a run on a 16 GB host does not
establish support for an 8 GB machine.

Model loading, first query execution, repeated query execution, and retrieval are
timed separately. First execution is not guaranteed to be cold in the OS disk
cache. These component timings are not interactive end-to-end search latency.
Vision throughput excludes session loading; OCR timings include preprocessing
and launching Tesseract. The first prototype holds generated embedding arrays in
memory before saving, so use modest corpus sizes; this is not the final indexer.

Four channels are reported: visual, lexical OCR, semantic OCR, and hybrid fusion.
Current fusion preserves cosine scores and gates a single OCR contribution per
image; historical reports used equal-vote reciprocal rank fusion.
Semantic passage matches are aggregated by image before fusion. Metrics include
Recall@5, Hit Rate@5, MRR@10, nDCG@10, and candidate Recall@50, grouped by query
language. Relevance is binary. Metrics use supplied positive judgments; missing
labels remain unjudged, so incomplete labels cannot establish true corpus recall.
When queries include a `category` field, results also include per-category
retrieval metrics.

To score OCR, pass `--ocr-truth data/ocr-truth.json`. The file is a JSON list of
objects with `image_id`, `language`, and `text` (the manually transcribed text).
The generated fixtures include this file. Reports include micro-averaged
character error rate (CER) and word error rate (WER), overall, by language, and
by image category when references include `category`.
Normalization uses Unicode NFKC, case folding, and collapsed whitespace;
punctuation is retained. Rates can exceed one when there are many insertions.
Empty reference denominators are undefined, not perfect scores.

OCR reports additionally count matching occurrences of Unicode word tokens,
including underscores, regardless of reading order. Token recall measures the
fraction of reference token occurrences recovered; token precision penalizes
extra output tokens. These metrics ignore punctuation, so they complement rather
than replace CER/WER. Optional reference `key_phrases` are tested as exact
substrings after the same Unicode/case/whitespace normalization, retaining
punctuation. Annotated key phrases must actually appear in the reference text.

INT8 is an experimental starting point. FP32 ONNX artifacts can be fetched and
benchmarked with `--precision fp32` on both commands (approximately 2.02 GB total
including tokenizers). Comparing those runs helps identify quantization losses;
it does not replace parity checks against the original framework. This harness
does not yet claim export parity or approve quantization for release.

## Offline verification

Benchmark processes use verified local paths, disable ONNX Runtime telemetry,
and block Python network operations with audit hooks. Python hooks do not block
native libraries or child programs at the OS level. Reports therefore leave
`os_network_isolation_verified` false by default.

For stronger verification, run the installed executable in an OS network sandbox
or disconnect networking after setup. Avoid `uv run` inside a network sandbox if
it would need to resolve dependencies. On macOS with `sandbox-exec` available:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python -m benchmarks run --data .cache/smoke-images --queries .cache/smoke-images/queries.json --corpus-kind generated-smoke --output runs/smoke-offline
```

Use equivalent network isolation on Linux or Windows; do not weaken existing
firewall rules. Record the isolation method alongside results rather than
assuming an environment variable proves offline operation.

## Development checks

```sh
uv run --no-sync black --check .
uv run --no-sync flake8 --config tox.ini src tests benchmarks
uv run --no-sync ruff check .
uv run --no-sync mypy src benchmarks
uv run --no-sync pytest
```

Black owns formatting; run `uv run --no-sync black .` to apply it. Flake8 reads
`tox.ini`, while Black and Ruff read `pyproject.toml`. All three use a 100-character
line length; Flake8 extends its default ignores with E203 for Black compatibility.
The Flake8 command targets source, tests, and benchmarks so local environments
and model files are outside its scan.
Ruff is used for linting rather than as a second formatter. The Python target is
3.12, matching the package's minimum supported version.

Tests use small local fixtures and require no model download. CI runs these
checks on macOS, Windows, and Linux. Full model/hardware benchmarks remain
explicit local runs.

For an editable development install, use `uv sync --locked`. If macOS marks the
environment's `.pth` files as hidden, Python may skip them and fail to import the
package. Use the regular installation above, or clear the hidden attributes on
those local environment files before using the editable install.

## Model references

- [SigLIP2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-224)
  and [pinned ONNX export](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX).
- [SigLIP2 preprocessing](https://huggingface.co/docs/transformers/model_doc/siglip2):
  lowercased text, EOS, fixed 64-token input; bilinear RGB resize and normalization
  matching the export's preprocessing configuration.
- [Multilingual E5 Small](https://huggingface.co/intfloat/multilingual-e5-small)
  and [ONNX export](https://huggingface.co/Xenova/multilingual-e5-small):
  `query: ` / `passage: ` prefixes, attention-mask mean pooling, L2 normalization.
- [Tesseract language data](https://tesseract-ocr.github.io/tessdoc/Data-Files.html).
  This benchmark uses your installed packs; report their provenance when sharing
  OCR results, since they may differ from the planned `tessdata_fast` distribution.

For an optional local alternative to Tesseract, see the
[OCR engine comparison guide](ocr-comparison.md).

Current OCR development evaluates [text-region detection with Tesseract](text-regions.md).
The [completed comparison](benchmarks/2026-10-19-tesseract-text-regions.md) covers
whole-image OCR, gating and experimental masking; it does not run GLM.

The [paired retrieval study](benchmarks/2026-10-19-text-gating-retrieval.md)
adds fresh text-presence controls and follows gating through the complete search
pipeline, including exact identifier diagnostics.
