# Incremental stage caching

Index refreshes scan for file changes and reuse successful analyses by image-content
SHA256 and stage dependencies. Adding one unique image processes that image; unchanged
images do not repeat OCR or model inference. Byte-identical copies share analyses.

## Dependencies

| Stage | Cache dependencies |
| --- | --- |
| Metadata / thumbnail | Image/preprocessing implementation, processing implementation, Pillow, thumbnail policy |
| OCR | Image/processing implementation, OCR implementation, NumPy/Pillow, languages, engine/runtime identity, selected language packs, orientation data, threads and storage budget |
| Vision | Image/processing implementation, selected SigLIP2 vision graph, NumPy/Pillow/ONNX Runtime and threads |
| Semantic OCR | OCR dependencies and actual consumed OCR text, processing/text encoder implementation, E5 graph/tokenizer, NumPy/ONNX Runtime/tokenizers, threads and passage budgets |

Changing OCR languages, engine or language packs invalidates OCR and semantic results,
while preserving compatible metadata and visual vectors. Changing SigLIP2 precision
in a separate collection reuses compatible OCR and E5 results. Existing collections
still require their saved vision precision; refreshes never silently convert them.
Shared processing/preprocessing source changes invalidate conservatively because
that code can affect several stages. Failed or skipped analyses are not reusable
successes. Changed OCR text cannot reuse stale semantic vectors, even when settings
are unchanged. Artifact checksums are checked before reuse; missing or corrupt
artifacts are processed again.

## Published indexes and older caches

The existing version-2 catalog schema and immutable full-pipeline manifests remain
compatible. Stage-key aliases reference validated existing artifacts; they do not
copy images, replace historical generations, or require a database migration.
The coordinator restores compatible stage records before starting each worker and
retains successful stage results for later configurations and collections.

Legacy full-pipeline records can be reused only when their recorded stage dependencies
match. Unknown or incomplete provenance is recomputed rather than guessed. The first
upgrade may repeat OCR and semantic processing because older records lack the new
orientation/engine provenance; compatible metadata and vision can still be reused.

Source and frozen Python implementations use shipped source hashes, avoiding false
changes from `.py` versus `.pyc` packaging. Existing recognized raw signatures remain
readable. Without shipped source, hashing falls back conservatively to the available
artifact. Bundled engine cache identity uses original binary hashes and TSV settings;
relocation, signing, manifest formatting, and license-notice changes do not invalidate
OCR. Package integrity verification still checks the distributed binary hashes.
System and bundled OCR runtimes are distinct unless their provenance establishes
compatibility; sharing a version string alone is insufficient.

## Progress and results

Live indexing progress reports the current stage's **processed**, **cached**, and
**total** image counts, with failures and skips shown separately. Cached images count
toward completed stage progress but are not presented as newly processed work.
The final indexing JSON includes `processed_stages` and `reused_stages`; the gallery
also summarizes OCR work at completion. These count stage attempts, including empty
semantic results, rather than individual model calls. Prior interrupted work reused
by a new job counts as cached for that new job.

Use `lens index FOLDER --index-dir WORKSPACE --log` for detailed per-stage logs.
See [validation](benchmarks/2026-10-24-stage-cache.md) for executed checks and limits.
