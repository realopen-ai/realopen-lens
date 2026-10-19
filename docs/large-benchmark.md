# Large image-pipeline benchmark

This suite measures execution reliability, OCR recovery, retrieval, latency, and
memory separately. Successful execution is not evidence of correct OCR or search.
The benchmark tools and annotations live under `benchmarks/`; application code
lives under `src/realopen_lens/`. Images, model weights, and run artifacts remain
in ignored local directories.

See [harness organization](benchmark.md#harness-organization) for the implementation
packages. The commands below remain supported compatibility entry points. Moving
harness files changes its source fingerprint: start a new run directory after
upgrading, or use the preserved original checkout to resume a historical run.
Reference assets, evaluation rules and retained result formats are unchanged.

## Prepare the collection

Install the development environment and pinned models using the
[benchmark setup](benchmark.md). Dataset preparation additionally requires `curl`
and Poppler's `pdftoppm` on `PATH`. PDF text references are already supplied; no
PDF library is required by the application or benchmark runner.

From the repository root:

```sh
uv run --no-sync python -m benchmarks.dataset --output data/benchmark-500-v3 --workers 3
```

The collection contains 500 distinct encoded images:

| Category | Images |
| --- | ---: |
| Photographs | 150 |
| Screenshots | 120 |
| Documents | 100 |
| Diagrams and charts | 80 |
| Controlled challenge variants | 50 |

There are 330 real source images, 120 authored synthetic cases, and 50 intentional
variants. Sources include individually licensed COCO photographs, Rich/Textual,
JabRef, JupyterLab, Streamlit, Tesseract test data, Mermaid, Matplotlib, C4 examples,
and IRS forms rendered from PDFs. Source notices and immutable Git object IDs
are retained in `benchmarks/sources/`. The catalog's legacy `commit` field holds
Git tree revisions returned by GitHub's tree API; downloaded blobs are verified
against their Git object hashes.

COCO selection uses official per-image license metadata and excludes photographs
with noncommercial/no-derivatives license IDs. Original Flickr source links and
license records remain in the manifest. Tesseract repository licensing is recorded
separately from unverified individual scan rights. Downloaded images are not
redistributed with the repository. Preserve applicable source attribution when
using the collection outside local evaluation.

The generator uses seed `20261019` and a checksum-verified, revision-pinned Noto
Sans font. Generated images use Pillow's basic layout engine. The exact corpus
used for published measurements is described by
`benchmarks/sources/dataset-v2.lock.json`. Pixel/encoding hashes can differ with
other Pillow/FreeType or Poppler versions; use the original dataset copy when an
exact cross-machine comparison is required. Every run verifies the local
manifest's image hashes before starting.

`manifest.json` includes category/subcategory, dimensions, format, source/license,
SHA-256, difficulty tags, expected decoding policy, OCR references where
available, queries, related-image families, and development/holdout assignments.
`near-duplicates.json` retains dHash review candidates. Blank margins and similar
page layouts create false positives; dHash is not a duplicate verdict. Exact
byte duplicates are rejected. Source and template families are intentionally
related and kept in the same split, as are derived variants. Known earlier
Rich/Tesseract diagnostic sources stay in development. Holdout includes unseen
sources/templates; it is not an independently annotated, blinded certification.

The set includes natural English/French text, one real Arabic image without a
transcript, and assistant-authored French/Arabic retrieval translations. These
translations are not independently reviewed. Arabic OCR quality and broad
language support remain unvalidated.

## Run and resume

Keep an unmodified engine checkout for baseline measurements. For example,
`baseline-checkout/` should preserve the original `src/` and `benchmarks/` directories. Run the same
suite, models, thread settings, references, and dataset for both implementations:

```sh
uv run --no-sync python -m benchmarks.suite --dataset data/benchmark-500-v3 --output runs/baseline --engine-root baseline-checkout/src --variant baseline
uv run --no-sync python -m benchmarks.suite --dataset data/benchmark-500-v3 --output runs/improved --engine-root src --variant improved
uv run --no-sync python -m benchmarks.compare --baseline runs/baseline --final runs/improved --output runs/comparison
```

The baseline variant calls the preserved original benchmark OCR worker, expected
at `baseline-checkout/benchmarks/worker.py`, and the original encoder/retrieval
package through `--engine-root`. Preserve both `src/` and `benchmarks/` when
making that baseline snapshot. To measure the current engine only, use
`--engine-root src --variant improved`.

For actual OS-level offline verification on macOS:

```sh
PYTHONPATH=src sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python -m benchmarks.suite --dataset data/benchmark-500-v3 --output runs/improved-offline --engine-root src --variant improved
```

Prepare dependencies, images, and models before running offline. Use equivalent
network isolation on other operating systems; Python audit hooks alone do not
block native networking.

Repeating a command with the same output directory resumes completed records.
A SQLite transaction commits each stage/image and its vectors together. Exceptions
and Tesseract timeouts are recorded per image and do not abort other images.
Use `--retry-errors` to retry failed stages explicitly. Configuration, dataset,
engine, harness, model, and environment mismatches require a new run directory;
previous results are not silently mixed. Source snapshots accompany each run.
An active-item heartbeat lets the supervisor isolate a native crash or hang:
it stops the job's process tree, records that item's failure, restarts the worker,
and continues. Startup/model-loading failures stop the job with a private log;
they are not fabricated as image failures. Defaults are 90 seconds per active
item and 120 seconds for startup, configurable with `--image-timeout` and
`--startup-timeout`. A whole-stage deadline also applies. Superseded/retried
records remain in `stage_history`, and memory profiles retain invocation history.

OCR has a 30-second recognition budget and at most four recognition attempts,
including any orientation retry. RGB is the primary input. Grayscale, narrow-rule
removal, and an opaque-region crop are fallbacks; a full-image attempt is retained.
Normal inputs fit 2400 × 2400 pixels, with a bounded 3200-pixel fallback for
large text-rich images. Word coordinates refer to `ocr_size`; `source_region`
and `rotation` describe their relationship to the oriented original. Source
decoding rejects more than 24 million pixels or 64 MiB of encoded data.
The benchmark workers run
sequentially with two inference threads by default. One model stage is resident
at a time. Semantic processing caps each image at 32,000 OCR characters and 16
passages, with an explicit warning when the budget is exceeded.

## Retained records and evaluation

Each run contains:

- `results.sqlite`: durable per-stage records and full embedding arrays.
- `individual.jsonl`: one complete record per image, including references, actual
  OCR, word boxes/confidence, timing, sampled RSS, warnings, errors, and verdicts.
- `queries.jsonl`: every query, its known positives, rankings, and metrics.
- `summary.json`: aggregate/category/subcategory, format, resolution, difficulty,
  and split metrics, plus retrieval metrics.
- `resources.json`: coordinator plus child-process RSS, including model loading.
- `failures.json`: flagged images with links to their diagnostic records.
- `run-events.jsonl` and stage logs: elapsed stage times and process diagnostics.
- `config.json`, `engine-source/`, `harness-source/`: reproducibility details.

Per-image RSS includes the resident stage model, not just the incremental cost
of that image. Global RSS sums processes, can double-count shared pages, and is
sampled every 20 ms; very short spikes can be missed. Access errors are recorded.
Stage-sum image latency excludes model loading, query encoding, and retrieval;
wall-clock stage timing is recorded separately. Neither is interactive search
latency. Running on a 16 GB host does not establish 8 GB support.

Full-transcript OCR references use CER/WER plus order-independent token recovery
and exact normalized key-phrase recall. References extracted from source PDF
text layers are identified explicitly. Visually reviewed partial references test
only their key phrases; they do not produce full-page CER/WER or token accuracy.
The diagnostic OCR verdict requires at least 80% recall/precision for available
token/phrase metrics. CER/WER remain reported but do not gate retrieval-oriented
OCR: diagrams and multicolumn pages can have several valid reading orders.
These thresholds are diagnostic, not product acceptance guarantees. Photo
subcategories are caption-based proxies, not reviewed scene labels. Synthetic
families share rendering components; the holdout is not an independent real-world
benchmark. Source-stratified metrics should accompany overall averages.

Images without a valid reference are `not_evaluated` for quality. Empty OCR is a
warning rather than proof that an image contains no text. Overall execution
success includes expected policy rejections; inspect their separate counts.
Retrieval reports visual, lexical, semantic OCR, and hybrid channels, using
Recall@5, Hit@5, MRR@10, nDCG@10, and candidate Recall@50. Known positives are
incomplete. Cropped variants are not assumed relevant when the query's subject
or text might have been cropped away. Natural, identifier, and translated
queries remain distinguishable.

The lexical index normalizes case/diacritics and preserves underscores. It can
use confidence-filtered observed OCR alternatives, with a lower contribution
than the canonical transcript; semantic embeddings and OCR accuracy metrics use
the canonical text. Fusion retains cosine scores, considers bounded candidates,
and uses corpus term frequency to distinguish rare anchors from common form
vocabulary. Correlated OCR signals contribute once. The fixed E5 gates were
calibrated on development judgments using `python -m benchmarks.calibrate`.
They are specific to the pinned export, not universal confidence probabilities.

Comparison output retains every image/query change, category and language
comparisons, and regressions. Dataset, queries, model artifacts, dependencies,
threads, language packs, and host identity must agree for comparison.

## Safety probes

Malformed files are additional tests, not inflated into the 500-image count:

```sh
uv run --no-sync python -m benchmarks.probes --output runs/input-probes
```

The nine cases cover truncated/empty files, HTML disguised as an image,
a decompression bomb, a valid file with a misleading suffix, multipage TIFF, and unsupported content disguised with an allowed suffix.
Results retain every case and distinguish snapshot RSS from sampled peak memory.

## Audited historical evaluation

The [hardening report](benchmarks/2026-10-19-pipeline-hardening.md) retains the
frozen v2 corpus and all raw runs. A post-run visual audit corrected a repository
basename collision and one mislabeled menu action. Corrections are explicit in
`benchmarks/sources/evaluation-corrections-v1.json`; all 500 images remain recorded.
One invalid query is retained in `excluded-queries.json`, outside the corrected
472-query denominator. Original 473-query reports are unchanged.

To apply the same corrections to an original run without inference:

```sh
uv run --no-sync python -m benchmarks.rescore --source runs/original --output runs/audited --corrections benchmarks/sources/evaluation-corrections-v1.json
```

Use a new output directory. Preconditions reject mismatched references. Comparisons
reject runs with different correction hashes. Newly built v3 manifests scope manual
annotations by exact repository and path and already contain the corrected labels;
do not apply the historical correction file to them. Published v2 scores require
the frozen manifest, not a newly built v3 corpus.
