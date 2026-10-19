# Persistent collection indexing and search

Lens can index a folder and search it across process restarts. This is the first
application workflow built on the evaluated inference and retrieval engine. A
gallery, API, image-to-image query command, and learned ranking remain planned.

## Setup and first collection

Use Python 3.12, install the package with uv, and install Tesseract 5 plus the
language packs you need. See the [setup guide](benchmark.md#setup) for native
installation and offline preparation. These commands run from the checkout:

```sh
uv sync --locked --no-editable --reinstall-package realopen-lens
uv run --no-sync lens doctor
uv run --no-sync lens models fetch
uv run --no-sync lens index data --collection personal --index-dir runs/personal-index
uv run --no-sync lens search "a dog on the beach" --collection personal --index-dir runs/personal-index
uv run --no-sync lens search "connection timeout" --mode lexical --collection personal --index-dir runs/personal-index
```

Only the explicit model-fetch commands download artifacts. Copy a prepared model
directory to use a machine entirely offline. Indexing and search require local,
checksum-verified INT8 models; missing files cause an error rather than a download.
Inference workers disable ONNX telemetry and deny Python networking. This is
defense in depth, not an OS network sandbox.

The default model directory is `models/` relative to the current working directory.
Set `--models /path/to/models` when running elsewhere. The default OCR language
selection is `eng+fra+ara`; use `--languages eng` or another installed selection
when indexing. Whole-image Tesseract is the collection workflow's only OCR path.
Experimental region detection, RapidOCR, and GLM-OCR are not enabled here.

JPEG, PNG, WebP, BMP, and TIFF are supported, including nested folders. Only the
first frame/page is indexed. PDFs must be rendered to supported image files first.
Inputs are bounded at 64 MiB encoded and 24 million decoded pixels. Original files
are read without modification; thumbnails use EXIF-oriented, white-composited RGB.

## Updating and recovery

Run the same `lens index` command again after adding, changing, moving, or deleting
images. Unchanged content reuses completed OCR, thumbnails, and embeddings. Exact
byte duplicates share one result with several paths. Renaming a file preserves
its SHA-256 image identity. Near-duplicate grouping is not implemented.

An existing collection name belongs to its original absolute root. Use a new name
if you need to index a different root. Multiple named collections can share an
index directory and reuse identical content. Directory symlinks are not followed;
file symlinks escaping the selected root or pointing into the index are reported
as failures. The index directory itself is excluded when placed inside the root.
Unreadable subdirectories fail the job rather than publishing false deletions.

By default, unchanged size, nanosecond modification/change timestamps, device,
and inode permit reuse of a prior content hash. `--rehash` forces hashing every
file. Processing rechecks source hashes, and publication rechecks file metadata,
so a file changed during a run is reported as an error. Search excludes paths
whose metadata has changed or which have disappeared; run indexing to update them.

```sh
uv run --no-sync lens status --collection personal --index-dir runs/personal-index
uv run --no-sync lens index data --collection personal --index-dir runs/personal-index --rehash
```

`status` shows the current phase, completed image count for that phase, current
paths, failed files, and the published generation. `index` prints its final JSON
summary on completion. Add `--log` for live, timestamped progress on stderr:

```sh
uv run --no-sync lens index data --index-dir runs/personal-index --log
# Save the JSON summary and progress independently:
uv run --no-sync lens index data --index-dir runs/personal-index --log > result.json 2> progress.log
```

Logs show discovery and cache counts, stage/model loading, per-image completion
times and failures, worker recovery/restarts, and publication. The default stays
quiet. Enabling logging does not invalidate cached processing or embeddings.
Native worker output remains saved in the job's local diagnostics file.

A corrupt input does not stop other images. Metadata/thumbnail or visual-embedding
failures exclude that image. An OCR failure or timeout instead leaves the image
searchable visually with `degraded` status. A semantic-embedding failure preserves
visual search and usable OCR keyword search. Missing text channels contribute no
OCR evidence to hybrid ranking. Empty OCR that completed successfully remains
distinct from failed OCR.

The final JSON distinguishes `failed_files` (excluded paths), `degraded_files`
(searchable paths missing text capabilities), and `degraded_images` (their unique
contents). `images` includes all searchable unique images. A run containing either
failed or degraded paths becomes `completed_with_errors` and exits with status 1.
Search results include `status` and structured `issues` identifying unavailable
stages; `status` lists failures and degradations separately. Fatal setup,
scan, or publication errors preserve the previous published generation.

Run the same indexing command to retry failed OCR or semantic embeddings. Successful
thumbnails, visual embeddings, and other completed stages are reused. Failed stages
are attempted once per unique image in each job, including when a native worker
crashes and restarts. Semantic processing is skipped when OCR is unavailable, and
is retried after OCR recovers. Existing error details remain visible until recovery;
retrying does not guarantee a difficult document will finish within the OCR budget.
`reused_images` counts images with every stage already complete; `reused_stages`
reports unique-image cache hits for each stage, including partial reuse during OCR
retries. When OCR is retried, its dependent semantic cache is invalidated so changed
text cannot retain stale passage embeddings.
Any processing retry publishes a fresh generation so updated OCR text, vectors,
and failure details become visible even if source paths and overall health did
not change. Runs that reuse every stage keep the existing healthy generation.

Only one indexing coordinator can write to a given index directory. Stage
workers process one image at a time with two CPU threads by default; `--threads`
accepts 1–8. Vision and semantic models run in separate processes, sequentially,
so their sessions are released between stages. A stage has a 120-second startup
deadline and a 90-second deadline per image. Native crashes and timeouts mark the
affected stage as failed and restart the worker for remaining images. OCR and
semantic failures degrade the image; metadata and vision failures exclude it. Detailed
worker diagnostics live in `logs/JOB_ID.log` under the index directory.

Press Ctrl-C to stop a CLI job. Completed stages are durable. If a coordinator is
killed abruptly, its stored job can still say `running` until the next indexing
command detects the interruption. The next run recovers the job, terminates its
identified stale stage process, rescans, and reuses completed work. This resumes
processing into a new job record, keeping the interrupted record for diagnosis.

## Search output and storage

Search modes are `hybrid` (default), `visual`, `lexical`, and `semantic`. Hybrid
uses the existing evidence-gated fusion of SigLIP2 visual similarity, literal OCR
matching, and multilingual E5 OCR passage similarity. It does not change the
ranking policy established by the earlier benchmarks. Lexical search loads no
neural model and does not require Tesseract. Other modes load their query encoders
sequentially in short-lived processes and verify compatibility with the index.
Cold queries therefore pay model startup costs; a resident API worker is future
work. Use `--limit` for 1–100 results.

Commands return JSON. Each result contains a stable content ID, matching original
paths, a local thumbnail path, oriented dimensions, an OCR snippet, and visual
similarity when computed. Similarity is a cosine score, not a calibrated relevance
probability. A missing or corrupt thumbnail is returned as null; reindexing repairs
it. Missing/corrupt search artifacts produce an explicit rebuild error.

Without `--index-dir`, Lens uses the OS application-data directory:

| Platform | Default directory |
| --- | --- |
| macOS | `~/Library/Application Support/RealOpenLens` |
| Windows | `%LOCALAPPDATA%/RealOpenLens` |
| Linux | `$XDG_DATA_HOME/realopen-lens`, falling back to `~/.local/share/realopen-lens` |

The directory holds sensitive derived data, including OCR text and source paths;
it is not encrypted. Its layout is:

```text
catalog.sqlite3            collections, files, jobs, stage results, active generation
analyses/PIPELINE/IMAGE/   thumbnails and per-image embedding cache
generations/GENERATION/   manifest, FTS5 database, vision/semantic NumPy arrays
logs/                     worker diagnostics by job
requests/                 temporary coordinator requests
indexing.lock             cross-process writer exclusion
```

The pipeline fingerprint includes pinned model artifacts, preprocessing code,
dependency versions, Tesseract/language-pack identity, languages, and resource
settings. Changed fingerprints invalidate cached processing. Vectors have explicit
image/passage row mappings and are loaded from disk-backed float32 arrays.
Upgrading from an earlier pipeline build can therefore require one full refresh;
ordinary retries within the same pipeline version reuse successful stages. Older
catalogs are migrated on the next indexing run, preserving existing records and
published generations. Search can still read the earlier catalog format.

Generations are built separately, checksummed, then published with one SQLite
transaction. Readers use one generation consistently while indexing proceeds.
Old generations and completed analysis caches are retained, including artifacts
for removed images. There is no automatic retention/garbage-collection policy yet.
To discard all derived data, stop Lens processes and delete the chosen index
directory; the original image folder is independent.

The engine's public entry points are `indexing.index_collection`,
`indexing.collection_status`, and `search.search_collection`. They return Python
dataclasses/dictionaries without depending on benchmark tooling, FastAPI, or React.

## Validation scope

Regression tests cover incremental updates, duplicate paths, source mutation,
corrupt artifacts, interrupted jobs, native timeouts/crashes, writer exclusion,
and failed publication. A real-model smoke run also exercises mixed photos,
screenshots, documents, and diagrams using Tesseract. This validates the storage
workflow; it does not establish new retrieval accuracy or 8 GB hardware support.
The 2,000-image collection target and native Windows/Linux installation still
need dedicated measurements. Existing quality results remain in the
[hardening report](benchmarks/2026-10-19-pipeline-hardening.md).

On the development Mac (ARM64, macOS 26.2, 16 GiB RAM), the real-model smoke set
contained 15 paths: 13 unique valid images, one exact duplicate, and one deliberate
corrupt file. The full processing run took 25.57 seconds inside the coordinator;
the corrupt file was reported while the other images became searchable. Subsequent
checks ran with all networking denied by `sandbox-exec`:

| Check | CLI wall time | Outcome |
| --- | ---: | --- |
| Unchanged collection | 0.98 s | 13 processed images reused; generation unchanged |
| One content edit, one rename, two removals | 2.02 s | 11 unchanged images reused; paths and new OCR updated |
| Forced hash verification | 0.51 s | 12 processed images reused; generation unchanged |
| Repair corrupt published vectors | 0.52 s | All 12 images rebuilt from cache; no inference repeated |
| Cold hybrid query | 1.89 s | Visual and semantic query encoders ran sequentially |
| Lexical query | 0.12 s | Expected screenshot returned for `LENS-7308` |

A separate eight-image run was interrupted by killing its coordinator during OCR.
The next run completed all eight images, marked the old job interrupted, and
preserved the already committed OCR payload unchanged. Regression tests also
exercise per-image crashes/timeouts and publication failures. The sampled peak
sum of process-tree RSS was about 921 MiB for the cold hybrid query and 665 MiB
for the incremental indexing check. RSS was sampled every 50 ms and may count
shared pages more than once. These are individual smoke measurements, not latency
percentiles, retrieval-quality estimates, or proof of an 8 GiB memory ceiling.
Local records, outputs, environment details, and the one-off validation script
are retained under the ignored `runs/collection-indexing/` directory.

The OCR degradation check used four forms that had previously timed out, a terminal
control, and a duplicate path (six paths, five unique images). With real Tesseract,
all completed successfully in 21.31 seconds on this rerun; the earlier timeout was
not reproduced. A separate, explicit fault-injection wrapper then made recognition
exit with code 17 while preserving normal version/language-pack checks. All five
unique images remained visually searchable, all six paths were reported degraded,
and the CLI returned structured JSON with exit code 1. Semantic OCR processing was
skipped without loading its encoder. After removing the injected fault, recovery
completed in 19.83 seconds with zero degraded paths; all ten metadata/vision cache
records were unchanged. These checks validate retention and retry behavior, not
an OCR accuracy or latency improvement. Local evidence is retained under ignored
`runs/ocr-degradation/`; regression tests also cover OCR timeouts, native crashes,
semantic failure, catalog migration, and stale text/vector invalidation.
