# SigLIP2 precision

Collection indexing supports `int8` (the default) and `fp32` SigLIP2. Both the
image encoder and visual query encoder use the selected precision. E5 semantic
text embeddings remain INT8; Tesseract and keyword search are unchanged.

## Try FP32 on your images

Install the current checkout with `uv sync --locked` after switching branches.
Explicitly fetch and verify the mixed model selection:

```sh
uv run --no-sync lens models fetch --siglip-precision fp32
uv run --no-sync lens models verify --siglip-precision fp32
```

These commands select FP32 SigLIP2 and INT8 E5. The older model setup option
`--precision fp32` selects FP32 for all encoders and is not required here.
Inference never downloads missing models automatically.

Build a separate collection index, leaving the original INT8 index available:

```sh
uv run --no-sync lens index data --index-dir runs/personal-index-fp32 \
  --siglip-precision fp32 --log
uv run --no-sync lens search "pizza" --index-dir runs/personal-index-fp32
uv run --no-sync lens search "pizza" --index-dir runs/personal-index
```

For a fair comparison, refresh the INT8 index against the same unchanged folder
and use matching OCR languages and thread settings. Different image inventories
or OCR configurations can also change results.

Search reads precision from the saved collection. There is no need to specify it
again. Optional `lens search --siglip-precision fp32` asserts that the collection
uses FP32; it rejects an INT8 collection rather than mixing incompatible embeddings.
`lens status --index-dir runs/personal-index-fp32` reports the published precision.

A collection's precision is immutable once published. Changing it requires a
different `--index-dir` or `--collection`, even for the same source folder.
The new collection runs the full indexing pipeline; cached analyses are not shared
between index directories. Each precision has its own processing fingerprint.
Existing collections without an explicit precision field are interpreted as INT8
and remain searchable when their existing model/runtime checks pass.

FP32 uses more memory and has higher model startup/query encoding costs in the
[earlier diagnostic study](benchmarks/2026-10-20-candidate-retrieval.md).
It is opt-in; this feature does not certify 8 GB hardware or native Windows/Linux
inference. Existing experimental rerankers and their training exporters require
INT8 SigLIP2 until a precision-aware training artifact is introduced.

## Python API

```python
from pathlib import Path
from realopen_lens.indexing import index_collection
from realopen_lens.search import search_collection

store = Path("runs/personal-index-fp32")
index_collection(Path("data"), index_dir=store, siglip_precision="fp32", log=True)
results = search_collection("pizza", index_dir=store)
```

The [production validation report](benchmarks/2026-10-20-siglip2-precision.md)
compares real persisted INT8 and FP32 collections, including cold-query memory.

## Production comparison

With the existing retrieval-v2 images and models available locally:

```sh
uv run --no-sync python -m benchmarks.experiments.siglip_precision \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --source-root data/retrieval-v2 --output runs/siglip2-precision-local
```

The runner selects two English query families per kind with seed 20261020,
retains all their reviewed labels and translations, and adds seeded distractors
to reach 128 images. This is a targeted integration diagnostic using previously
reviewed judgments. Its scores are not comparable with the full 2,052-image study
and do not constitute a fresh holdout. Configure `--image-count`,
`--families-per-kind`, `--threads`, and `--languages` as needed.

Both policies build separate production indexes and execute complete production
hybrid queries. Output retains frozen subset manifests, each indexing attempt,
sampled process-tree memory, SQLite/JSONL query records, quality metrics,
latency distributions, source snapshots, and `comparison.json`.
Degraded images remain recorded. A missing image or failed query fails the run.

Repeat the identical command to reuse indexed images and resume evaluation.
Initial indexing timing remains in `index-initial.json`; subsequent indexing
attempts are saved separately. Changing configuration, source, models, or inputs
requires a new output directory. Memory is sampled RSS, including model loading;
shared pages may count twice and brief peaks can be missed. Sequential runs are
subject to cache and background-load differences.

Check `memory_measurement.complete` in `comparison.json` before interpreting RSS.
Nonzero access errors mean child-process memory could not be sampled completely.

For an existing full collection, use the [saved collection evaluation workflow](saved-precision-evaluation.md)
to verify reference membership, retain additional distractors, and build a matched
INT8 control without repeating the successful OCR/E5 stages.
