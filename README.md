# RealOpen Lens

Private, offline image search for your own collection, by RealOpen AI.

RealOpen Lens finds photos, screenshots, diagrams and document images using visual
content, visible text and retained metadata. Your originals stay in place; indexing,
search, feedback and organization run locally.

**Release version: 0.1.0.** The core includes a desktop launcher, local gallery,
resumable indexing, multilingual OCR, hybrid text/image search and collection
management. Optional experimental features are identified below. See the
[0.1.0 release notes](docs/releases/v0.1.0.md) and [changelog](CHANGELOG.md).

## Install and start

Download the archive for your platform and its checksum from
[GitHub Releases](https://github.com/realopen-ai/realopen-lens/releases).

| Platform target | Archive | Instructions |
| --- | --- | --- |
| macOS, Apple silicon (arm64) | `RealOpenLens-macos-arm64.zip` | [macOS desktop](docs/desktop.md#first-run-setup) |
| Windows, x64 | `RealOpenLens-windows-x64.zip` | [Windows desktop](docs/desktop-windows.md#starting-lens) |
| Ubuntu 24.04, x64 graphical desktop | `RealOpenLens-linux-x64.tar.gz` | [Linux desktop](docs/desktop-linux.md#install-and-start) |

Extract the complete application, open the launcher, then use **Device setup** to
choose existing models or explicitly download them. Open the image library and
select **Add folder**. You can select folders anywhere your OS permissions allow;
they do not need to be inside this repository's `data/` directory.

The archives include Python, the gallery, Tesseract, 19 OCR language packs and
orientation data. Python, Node.js and a separate OCR installation are not needed
on the target device. Model weights are provisioned separately. Once setup is
complete, indexing and search work offline. The macOS app is ad-hoc signed and
unnotarized; Windows and Linux archives are unsigned. See the platform guides for
startup requirements and OS security prompts.

For source/CLI installation, use the [collection guide](docs/collections.md).
Benchmark contributors can use the [benchmark guide](docs/benchmark.md) and
[500-image study](docs/large-benchmark.md). The root `data/` directory is an ignored
convenience for development, not an application requirement.

```sh
uv run --no-sync lens index /path/to/images --index-dir runs/personal-index --log
uv run --no-sync lens status --index-dir runs/personal-index
uv run --no-sync lens search "database connection error" --index-dir runs/personal-index
```

## Everyday workspace

Keep recorded searches, relevance labels, virtual collections and duplicate reviews
together in a [persistent workspace](docs/workspace.md). Existing matching indexes
can be previewed and imported without modifying their originals. After setup,
`uv run --no-sync lens serve` opens the default workspace on port 8765.

## Local gallery

Browse collections, search by text or image, inspect results, and give explicit
relevance feedback in the [local React gallery](docs/local-gallery.md). Indexing
and search tasks stream live progress through SSE. [Search pagination](docs/search-pagination.md)
lets you load more text and similarity results from a stable, bounded pool. The [image viewer](docs/image-viewer.md)
provides zoom, keyboard navigation, full indexed OCR text and collection controls.
[Natural-language metadata search](docs/metadata-search.md) adds camera/date/file
constraints and offline GPS place lookup, including HEIC/HEIF photo support.
The [timeline gallery](docs/timeline-gallery.md) groups images by month and day,
with date sorting and scrolling across folders and virtual collections.

```sh
cd frontend
npm ci
npm run build
cd ..
uv sync --locked --extra web --no-editable --reinstall-package realopen-lens
uv run --no-sync lens serve --index-dir runs/personal-index --allow-root /path/to/images
```

Open **http://127.0.0.1:8765**. Use your existing index directory; a fresh directory
starts an empty library. Models and Tesseract language packs must be installed
before indexing when running from source. Desktop bundles include
Tesseract with 19 OCR language packs and orientation data. Compiled assets run
locally without a Vite server or internet.

## Features

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
Desktop bundles include [19 OCR language packs](docs/multilingual-ocr.md), selected
through named manual choices (English/French/Arabic by default) or optional local
automatic script-group detection. Synthetic diagnostics cover every pack;
independent real-world language review remains incomplete.

## Local by design

Images, thumbnails, extracted text, embeddings, queries, and feedback stay on your
device. The application runs without cloud inference, accounts, or telemetry.
Initial setup can download model files explicitly; offline model bundle import is
also planned. Once setup is complete, indexing and search work without a network
connection.

Original images remain in their existing folders. Lens stores its index and
derived data separately.

## Built for consumer hardware

Lens uses CPU inference; Docker and a dedicated GPU are not required. New
collections default to INT8 SigLIP2 below 16 GiB physical RAM and FP32 at 16 GiB
or above, with E5 INT8 in both profiles. Saved preferences and explicit choices
are respected. Memory and latency depend on image sizes, OCR languages and the
selected model profile; see the [precision guide](docs/siglip2-precision.md).

Native release acceptance exercises packaged INT8/FP32 inference through GitHub's
macOS 15 arm64, Windows Server 2022 x64 and Ubuntu 24.04 x64 runners. These are the
validation hosts; they do not establish compatibility with every OS version,
architecture or Linux distribution. Ubuntu's desktop/browser/fonts remain host
requirements. See the [release scope and limitations](docs/releases/v0.1.0.md#scope-and-limitations).

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

## License

Project source is licensed under [Apache-2.0](LICENSE). Third-party dependencies,
downloaded model weights and OCR language data retain their own licenses.
See [third-party notices](THIRD_PARTY_NOTICES.md) for the desktop dependency
inventory, copyright notices and license texts.

## Release scope and future work

The 0.1.0 core includes:

- Resumable indexing, incremental stage caches and live progress/cancellation.
- Visual, lexical and semantic OCR retrieval with hybrid fusion.
- Folder, library-wide and virtual-collection text/image search, with pagination.
- EXIF/HEIF metadata, offline GPS place lookup and natural-language constraints.
- Timeline browsing and an image viewer with OCR, metadata and source actions.
- Explicit relevance feedback, exact-query memory and reviewed local training controls.
- Folder refresh/rename/removal, virtual collections and duplicate review.
- macOS arm64, Windows x64 and Ubuntu 24.04 x64 desktop packages with bundled OCR.
- Reproducible evaluations and native package/model acceptance workflows.

Optional automatic script-group OCR, learned reranking, mixed-precision library
ranking and text-region OCR remain experimental. Hybrid ranking and manually
selected whole-image Tesseract remain the default paths. Broader independent
multilingual review, a neural reranker, saved-search shortcuts and expanded OS
coverage are future work, not required for this release.

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
├── api/            # Local HTTP adapter, tasks and progress streams
├── desktop/        # Native launcher, device setup and model provisioning
├── indexing/       # Discovery, incremental orchestration, processing stages
├── encoders/       # CPU ONNX encoders and image preprocessing
├── ocr/            # Tesseract, backend selection, experimental alternatives
├── metadata/       # EXIF, capture dates, offline places and query constraints
├── organization/   # Folders, virtual collections and duplicate groups
├── retrieval/      # Ranking, fusion, and saved-collection search
├── learning/       # Explicit reranker features and local JSON inference
├── storage/        # SQLite catalog, caches, immutable search generations
├── runtime/        # Worker processes, supervision, offline enforcement
├── models/         # Artifact management and pinned manifests
├── workspace/      # Persistent workspace identity and imports
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

New collections use a shared [device precision policy](docs/siglip2-precision.md):
INT8 SigLIP2 below 16 GB physical RAM, FP32 at 16 GB or above, with E5 INT8.
Saved preferences and explicit overrides are respected; existing indexes retain
their recorded precision.
See the [production validation](docs/benchmarks/2026-10-20-siglip2-precision.md)
for measured quality, cold-query latency, and memory.
The [full saved-collection comparison](docs/benchmarks/2026-10-20-full-siglip2-precision.md)
evaluates all 357 frozen queries against 2,227 searchable images, retaining first-pass
failures and retries. Use the [saved precision guide](docs/saved-precision-evaluation.md)
to build a matched control and reproduce the evaluation on your collection.

Find [visually similar images](docs/image-similarity.md) using a local image or an
indexed image ID:

```sh
uv run --no-sync lens similar /path/to/query.jpg --index-dir runs/personal-index
uv run --no-sync lens similar --image-id FULL_IMAGE_ID --index-dir runs/personal-index
```

Exact byte matches reuse saved vectors; the query itself is excluded by default.

Save a search for [explicit local relevance feedback](docs/relevance-feedback.md):

```sh
uv run --no-sync lens search "pizza" --index-dir runs/personal-index --save-run
uv run --no-sync lens feedback runs --index-dir runs/personal-index
```

Label returned results as relevant or irrelevant, correct/remove labels, and
export them as JSONL. With exact-query memory enabled, relevant images are promoted and irrelevant images
are hidden for that exact query and scope. Feedback never automatically trains or
activates a model. See [relevance feedback](docs/relevance-feedback.md).

For repeated queries, an explicit [resident search session](docs/resident-inference.md)
reuses query encoders through the Python API or `lens search --interactive`.
Single-shot search remains available; sessions require explicit ownership and shutdown.

Hybrid search recognizes conservative [typed numeric identifiers](docs/numeric-identifiers.md)
such as form, invoice and error numbers across supported marker languages, preserving
full codes and leading zeros. See the
[before/after evaluation](docs/benchmarks/2026-10-20-numeric-identifiers.md) for measured gains and limits.

The [candidate retrieval study](docs/benchmarks/2026-10-20-candidate-retrieval.md)
compares pool depths, encoder precision, reviewed query expansion and OCR timeout
recovery. See the [reproduction guide](docs/candidate-retrieval.md) for controls and commands.

The [500-image hardening report](docs/benchmarks/2026-10-19-pipeline-hardening.md)
includes before/after results, failure analysis, resource costs and remaining limitations.

The [text-gating retrieval study](docs/benchmarks/2026-10-19-text-gating-retrieval.md)
evaluates downstream search and exact identifiers on the historical corpus plus
fresh text-presence and small/faint-text cases.

Reviewed feedback can also train a versioned local logistic reranker, with explicit
activation and rollback. See [feedback training](docs/feedback-training.md).

[Virtual collections](docs/virtual-collections.md) let you save named image selections,
browse them and search within them while keeping source files in place.

See [duplicate groups](docs/duplicate-groups.md) for exact pixel result collapse,
local near-duplicate review and resumable scans.
