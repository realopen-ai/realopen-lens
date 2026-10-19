# RealOpen Lens architecture

Status: accepted direction, 2026-10-18.

## Product scope

RealOpen Lens is a standalone, private, offline image search and organization
application. Its first release targets macOS, Windows, and Linux, including
CPU-only consumer machines with 8 GB RAM. The 16 GB profile improves throughput
and model residency rather than enabling essential features unavailable at 8 GB.

Multilingual retrieval is part of the first release. Supported and tested
languages must be documented separately from the language coverage advertised by
upstream models. OCR uses explicitly selected local language packs.

## Application boundaries

- A reusable Python engine owns ingestion, retrieval, ranking, feedback, and
  evaluation. It is independent of the API and UI frameworks.
- A CLI and a thin FastAPI adapter invoke the same application operations.
- A React interface uses the local API. Production serves compiled assets locally.
- A managed worker subprocess performs indexing and inference. Model instances
  are not duplicated across API workers. Worker shutdown releases model memory.
- SQLite stores metadata, OCR, job progress, collections, and relevance feedback.
  Local embedding files support vector retrieval.

Use narrow interfaces at actual replacement boundaries: encoders, OCR, metadata
storage, lexical retrieval, and vector retrieval. Keep storage rows, SQL dialects,
and web request objects out of the engine's public data types. Avoid a generic
plugin framework or speculative implementations.

## Selected technologies

| Area | Decision |
| --- | --- |
| Language | Python 3.12 |
| Runtime inference | ONNX Runtime with CPU as the supported baseline |
| Visual and query embeddings | `google/siglip2-base-patch16-224` |
| OCR | Tesseract 5 with selected `tessdata_fast` language packs |
| OCR semantic embeddings | `intfloat/multilingual-e5-small` |
| Metadata and durable job state | SQLite |
| Lexical retrieval | SQLite FTS5 |
| Initial vector retrieval | NumPy exact cosine search over disk-backed arrays |
| Image processing | Pillow |
| API | FastAPI and Uvicorn |
| UI | React, TypeScript, Vite |
| CLI and development tooling | Typer, `pyproject.toml`, uv |
| Optional training dependencies | scikit-learn and PyTorch |
| Quality checks | pytest, Black, Flake8, Ruff, type checking, native OS CI |

Pin model revisions, tokenizer files, preprocessing, ONNX exports, and checksums.
Validate exports against reference outputs before distribution. Quantization and
hardware acceleration require measured quality and performance checks; neither
is assumed to be faster on every platform. Training and export tools remain
optional dependencies outside the normal inference installation.

Whole-image Tesseract remains the default OCR path.
[Text-region gating and masking](text-regions.md) are experimental, explicitly selected
paths. Their measured latency, memory cost, and recall risks do not justify promotion.
The RapidOCR and GLM-OCR comparisons remain [documented experiments](ocr-comparison.md).
Current pipeline evaluation uses Tesseract only. Promotion requires measured
quality and resource results. Add FAISS only when measured collection size and
latency justify an additional search backend.

## Indexing and search

Indexing scans user-selected folders without modifying originals, assigns stable
image identities, extracts metadata, generates thumbnails, performs OCR, and
computes embeddings. Store OCR passage embeddings separately from image vectors.
Track changes, missing files, and moved files. Persist stage progress so failed
or interrupted jobs can resume.

Version embeddings by model artifact and preprocessing configuration. Maintain
explicit mappings from vector rows to image or passage identifiers. Publish
index generations consistently with database state and recover from partial
writes without returning mismatched records.

The [persistent collection CLI](collections.md) implements these operations with
a SQLite catalog, content-addressed stage cache, immutable FTS5/vector generations,
and a cross-process writer lock. Search reads one published snapshot; indexing
publishes a replacement only after its artifacts are complete.
Metadata and visual embeddings are required for publication. Failed OCR or semantic
OCR embeddings degrade the image while preserving usable search channels. Failed
text stages retry on later indexing jobs without repeating successful image stages.

Search retrieves candidates independently through visual similarity, lexical
OCR matching, and semantic OCR similarity. Aggregate passage matches into image
results before fusion. Preserve similarity scores and keep one bounded OCR
contribution per image, so correlated lexical and semantic signals do not
automatically outweigh strong visual matches. Weak OCR evidence leaves visual
ordering unchanged. Fixed gates are versioned with their model export and
validated against held-out data; equal-vote reciprocal rank fusion remains a
historical benchmark baseline. Apply filters consistently with the
candidate search so filtered queries do not silently lose relevant candidates.
An optional learned reranker reorders the resulting pool.

Start ranking experiments with fixed fusion, logistic regression with query-score
interaction features, and a small neural network. Deploy a trained reranker only
when held-out evaluation supports it. Training runs explicitly in bounded jobs;
feedback does not trigger retraining after every interaction.

## Resource behavior

The 8 GB profile uses small batches, a single indexing worker, bounded image
decoding and OCR inputs, and one resident inference model at a time. Run indexing
in stages to avoid repeatedly loading models for individual files. Search may
load query encoders sequentially. The 16 GB profile may retain models and use
larger batches when measurements permit.

Use the same model artifacts across profiles initially; changing batch sizes or
residency policy must not require rebuilding the collection. Do not treat
memory-mapped arrays as free memory: searches still read data and populate the
OS cache.

An initial acceptance target is less than 2 GB peak application memory for the
8 GB profile on a defined benchmark collection, including child processes.
This is a target to validate on real hardware, not an established measurement.
Report collection size, OCR passage counts, cold and warm latency, indexing
throughput, memory, and retrieval quality together.

## Privacy and offline operation

All images, derived data, queries, feedback, and model updates remain local.
Runtime loads models only from local paths. Network access is confined to an
explicit setup/download operation, with offline bundle import supported.
Bundle frontend assets and fonts. Do not add telemetry or remote inference.

Bind the standalone API to loopback, validate browser origins, and protect local
API access. Limit file access to selected collection roots and controlled app
storage. Treat OCR, thumbnails, embeddings, and feedback as private derived data.
Do not log image contents or queries by default.

Verify indexing, search, feedback, and restart recovery with outbound networking
blocked. Cover native path handling and installation on all three operating
systems without requiring Docker or WSL for standalone Lens.

## Evaluation and initial implementation gate

Before building the full UI, validate SigLIP2 ONNX inference, multilingual OCR
quality, and memory behavior on an actual 8 GB machine. Evaluate visual,
lexical, semantic OCR, and hybrid baselines separately, with per-language and
per-image-category results. Hold out query families and duplicate image groups;
keep a final test set separate from feedback-based model selection.

Measure candidate recall independently of final ranking. Distinguish Recall@K
from Hit Rate@K, and unjudged candidates from confirmed negatives. A feedback
demonstration on corrected queries and generalization to untouched queries are
separate results. Version and retain promoted rerankers so updates can roll back.
