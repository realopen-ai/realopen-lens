# RealOpen Lens

Private, offline image search for your own collection.

RealOpen Lens is being built to help you find photos, screenshots, diagrams, and
document images by their visual content and the text they contain. It combines
multilingual image embeddings, optical character recognition (OCR), and hybrid
retrieval, with a lightweight ranking model that can learn from explicit feedback.

> **Status: early development.** Persistent folder indexing and CLI search are
> available, alongside inference and retrieval benchmarks and an opt-in experimental
> logistic reranker. The gallery, API, feedback, and neural ranking are still planned.

Start with the [collection indexing and search guide](docs/collections.md), the
[benchmark setup and usage guide](docs/benchmark.md), or the
[500-image benchmark guide](docs/large-benchmark.md). Add your own
images to the root `data/` folder, which is ignored by Git.

After [installing the package, Tesseract, and local models](docs/collections.md#setup-and-first-collection):

```sh
uv run --no-sync lens index data --index-dir runs/personal-index --log
uv run --no-sync lens status --index-dir runs/personal-index
uv run --no-sync lens search "database connection error" --index-dir runs/personal-index
```

## What you'll be able to do

- Search images using natural-language descriptions in multiple languages.
- Find screenshots by error messages, code snippets, and other visible text.
- Search extracted text by meaning, including across languages.
- Use an image to find visually similar images.
- Browse results through a local interface or search from the command line.
- Group near-duplicates and organize virtual collections without moving originals.
- Mark results relevant or irrelevant and evaluate locally trained ranking updates.

Example searches:

> “Sunset photos near the ocean”
>
> “The screenshot with a PostgreSQL connection error”
>
> “Un schéma expliquant les réseaux de neurones”

These examples illustrate intended behavior, not measured retrieval results.
Tested language coverage will be published as evaluation progresses.

## Local by design

Images, thumbnails, extracted text, embeddings, queries, and feedback stay on your
device. The application will run without cloud inference, accounts, or telemetry.
Initial setup can download model files explicitly; offline model bundle import is
also planned. Once installed, indexing and search will work without a network
connection.

Original images remain in their existing folders. Lens stores its index and
derived data separately.

## Built for consumer hardware

The first release targets **macOS, Windows, and Linux**, with **CPU-only operation
and 8 GB RAM** as the minimum hardware goal. A 16 GB profile will allow larger
indexing batches and more resident models. Docker and a dedicated GPU will not be
required.

Hardware support, memory use, and performance are targets awaiting validation on
real machines. Language support will be evaluated separately for visual search,
OCR, and text retrieval.

## How it works

1. **Index:** scan selected folders, fingerprint images, extract metadata, generate
   thumbnails, read visible text, and compute embeddings.
2. **Retrieve:** search visual embeddings, OCR keywords, and OCR passage embeddings
   independently.
3. **Combine:** merge candidates using score-preserving, evidence-gated fusion and optionally
   rerank them with a validated learned model.
4. **Improve:** store explicit feedback locally and compare candidate ranking
   updates against held-out evaluation data before promotion.

The reusable Python engine sits behind a CLI and a thin FastAPI adapter. A React
interface provides the local gallery, while a managed worker handles indexing and
inference with bounded resource use.

| Component | Selected technology |
| --- | --- |
| Engine and inference | Python 3.12, ONNX Runtime |
| Visual embeddings | SigLIP2 Base Patch16, 224 px |
| OCR | Tesseract 5 with selected language packs |
| OCR semantic embeddings | Multilingual E5 Small |
| Storage and retrieval | SQLite, FTS5, NumPy |
| API and interface | FastAPI, React, TypeScript, Vite |
| Reranker training | NumPy logistic baseline; neural training deferred |

See [the architecture document](docs/architecture.md) for model choices, resource
profiles, privacy requirements, and evaluation methodology.

Whole-image Tesseract is the default OCR path.
[Text-region gating and masking](docs/text-regions.md) are experimental and require
explicit selection; the retrieval study did not establish an overall search benefit.
The earlier [OCR engine experiments](docs/ocr-comparison.md) remain documented;
current pipeline evaluation uses Tesseract only.

## Roadmap

- [ ] Validate multilingual inference, OCR quality, and memory use on 8 GB hardware.
- [x] Build the Python package, CLI, and resumable ingestion pipeline.
- [x] Implement visual, lexical, and semantic OCR retrieval with hybrid fusion.
- [x] Publish reproducible baseline evaluations by language and image category.
- [x] Train and evaluate an experimental logistic reranker against hybrid fusion.
- [ ] Expand independent relevance review and compare a small neural reranker.
- [ ] Add explicit feedback, validated model updates, and collection management.
- [ ] Build the local image gallery and verify installation on all three platforms.

See [the collection guide](docs/collections.md) for installation and CLI usage, and
[the benchmark guide](docs/benchmark.md) for offline setup, query judgments, and
measurement limitations.

## Repository layout

- `src/realopen_lens/`: application engine and CLI.
- `benchmarks/`: development harness, sample downloaders, fixtures, and evaluation metrics.
- `tests/`: automated tests.
- `docs/`: architecture, benchmark instructions, and published results.

The engine is grouped by responsibility:

```text
src/realopen_lens/
├── indexing/       # Discovery, incremental orchestration, processing stages
├── encoders/       # CPU ONNX encoders and image preprocessing
├── ocr/            # Tesseract, backend selection, experimental alternatives
├── retrieval/      # Ranking, fusion, and saved-collection search
├── learning/       # Explicit reranker features and local JSON inference
├── storage/        # SQLite catalog, caches, immutable search generations
├── runtime/        # Worker processes, supervision, offline enforcement
├── models/         # Artifact management and pinned manifests
├── cli.py
├── search.py       # Stable public search entry points
└── __init__.py
```

The documented Python entry points remain `realopen_lens.indexing.index_collection`,
`realopen_lens.indexing.collection_status`, and `realopen_lens.search.search_collection`.

Benchmark tools run from the checkout with `python -m benchmarks`; they are not
included in the application wheel. See the [benchmark guide](docs/benchmark.md)
for commands and setup.

Benchmark implementations are grouped into `datasets/`, `metrics/`, `runtime/`,
`storage/`, `reporting/`, and `experiments/`. Existing top-level command modules
remain as thin compatibility entry points. Pinned sources and OCR reference
assets retain their original locations.

Evaluate published collections and export reviewed candidate features using the
[saved-collection retrieval guide](docs/retrieval-evaluation.md). The
[2,000-image retrieval report](docs/benchmarks/2026-10-20-retrieval-evaluation.md)
records the production baseline, reviewed-label limitations, and training exports.

The [experimental logistic reranker](docs/logistic-reranker.md) learns from reviewed
training pairs and is selected explicitly with `lens search --reranker MODEL.json`.
Hybrid remains the default. Its [initial comparison](docs/benchmarks/2026-10-20-logistic-reranker.md)
records both ranking improvements and text-heavy retrieval regressions.
The [fresh-source scope study](docs/benchmarks/2026-10-20-reranker-scope-study.md)
adds reviewed document/screenshot coverage, feature ablations and a protected
top-50 reranking scope, with individual results and remaining recall limitations.

Collection indexing now supports opt-in [FP32 SigLIP2](docs/siglip2-precision.md)
with matching query encoders and a separate index; INT8 remains the default.
See the [production validation](docs/benchmarks/2026-10-20-siglip2-precision.md)
for measured quality, cold-query latency, and memory.

The [candidate retrieval study](docs/benchmarks/2026-10-20-candidate-retrieval.md)
compares pool depths, encoder precision, reviewed query expansion and OCR timeout
recovery. See the [reproduction guide](docs/candidate-retrieval.md) for controls and commands.

The [500-image hardening report](docs/benchmarks/2026-10-19-pipeline-hardening.md)
includes before/after results, failure analysis, resource costs and remaining limitations.

The [text-gating retrieval study](docs/benchmarks/2026-10-19-text-gating-retrieval.md)
evaluates downstream search and exact identifiers on the historical corpus plus
fresh text-presence and small/faint-text cases.
