# Explicit relevance feedback

Save a search, then mark individual results relevant or irrelevant. Feedback stays
in a local SQLite database. It does not change search ordering or trigger training.
Text search, image similarity and resident text-search sessions are supported.

## Save a search

```sh
uv run --no-sync lens search "pizza" --index-dir runs/personal-index --save-run
```

The command keeps the normal JSON result list on stdout and prints
`Saved search: RUN_ID` on stderr. Without `--save-run`, searches are not recorded.
Recording an empty result list is allowed; it creates no judgments.

The same option works for image queries and interactive searches:

```sh
uv run --no-sync lens similar /path/to/query.jpg \
  --index-dir runs/personal-index --save-run

uv run --no-sync lens search "pizza" \
  --index-dir runs/personal-index --interactive --save-run
```

Each interactive query gets its own saved run ID. Precision and ranking options
continue to work normally. Recording does not load an additional encoder.

## Review and label results

Use the run ID printed by the search and an `image_id` from its results:

```sh
uv run --no-sync lens feedback runs --index-dir runs/personal-index
uv run --no-sync lens feedback show RUN_ID --index-dir runs/personal-index

uv run --no-sync lens feedback set RUN_ID IMAGE_ID relevant \
  --index-dir runs/personal-index

uv run --no-sync lens feedback set RUN_ID IMAGE_ID irrelevant \
  --index-dir runs/personal-index
```

`set` corrects an existing judgment. Repeating the same label is idempotent.
Only `relevant` and `irrelevant` are accepted, and only images actually returned
by that saved search can be judged. Byte-identical source copies share one
content ID and one result judgment within a run.

Remove a judgment to return the pair to an unlabeled state:

```sh
uv run --no-sync lens feedback remove RUN_ID IMAGE_ID --index-dir runs/personal-index
uv run --no-sync lens feedback history RUN_ID --index-dir runs/personal-index
```

Corrections and removals append timestamped revisions to an audit history.
Removing a label removes it from current judgments and future exports; the
previous labels remain in history. It is not a history-erasure operation.

Judgments apply to the saved display. They remain accessible after reindexing,
removing an original image, or deleting its old generation artifacts. A newer
search creates a separate run; it does not silently replace earlier judgments.

## Export labels

```sh
uv run --no-sync lens feedback export runs/feedback-labels.jsonl \
  --index-dir runs/personal-index
```

The command prints the number of exported records. It writes UTF-8 JSONL from a
consistent database snapshot, publishing the completed file atomically. Existing
exports are preserved unless `--overwrite` is explicitly supplied. Default
publication uses a same-directory hard link; filesystems without hard-link
support return an error and leave the stored feedback intact.

Each record contains:

- The query text, or the image query's SHA256 ID; returned image ID.
- Explicit judgment and numeric label (`relevant=1`, `irrelevant=0`), revision and timestamps.
- Collection identity, original generation and manifest checksum.
- Encoder specification, pipeline identity, ranking configuration and ranking source hashes.
- Original result rank, visual similarity, dimensions and processing status.
- Raw hybrid candidate scores/ranks when that result has a captured hybrid candidate.

Exports include only the latest active explicit judgments. Unjudged and removed
results are omitted. The exporter does not infer negatives from non-clicks,
non-returned images or missing labels. Original paths and OCR previews are not
included in exports; query text itself may still contain private information.

Image, lexical, semantic and visual-only runs do not have hybrid candidate
features, so their `candidate` and `candidate_schema` are null. Hybrid candidates
use raw candidate schema version 1. Custom candidate depths are preserved; they
must be checked against a future trainer's feature constraints before use.

This is a versioned feedback export, not a drop-in replacement for frozen
benchmark judgments or existing training artifacts. It has no automatic
train/validation/test assignment. Before training, review conflicting judgments,
group related queries and source/duplicate images, and keep a separate untouched
evaluation set. Multiple judgments across different runs are intentionally
retained rather than silently deduplicated.

## Python API

```python
from pathlib import Path
from realopen_lens.learning.feedback import (
    search_and_record, judge, remove_judgment, export_feedback,
)

index = Path("runs/personal-index")
recorded = search_and_record("pizza", index_dir=index)
if recorded.results:
    image_id = recorded.results[0].image_id
    # Call only after the user explicitly confirms the label.
    judge(recorded.run_id, image_id, "relevant", index_dir=index)
    remove_judgment(recorded.run_id, image_id, index_dir=index)

export_feedback(Path("runs/feedback-labels.jsonl"), index_dir=index)
```

Other entry points in the same module are `similar_and_record`, `saved_runs`,
`saved_search`, and `judgment_history`. For resident encoders, use
`SearchSession.search_recorded(...)`; it returns a `RecordedSearch` containing
`run_id` and the normal `SearchResult` list. `saved_runs` lists up to 100 recent
runs. The recording wrappers own their diagnostic trace.

## Storage and provenance

Opt-in searches and judgments live in `feedback.sqlite3` beside `catalog.sqlite3`
in the index directory. Existing collection catalogs and their schema are
unchanged. Readers use read-only connections; mutations use SQLite transactions
and serialized judgment updates. Feedback storage has its own schema version and
rejects unrelated or unsupported databases.

A saved run contains the original query, returned result metadata (including
source paths and OCR previews), candidate evidence for returned hybrid results,
encoder specification, effective precision/budgets, reranker checksum/scope,
and ranking source hashes. Image queries store their content ID rather than the
external query path. If the active collection changes during recording, the
operation fails and asks for a retry rather than associating results with the
wrong generation. Failed searches are not saved.

No relevance feedback is inferred automatically. The system has one local
judgment stream, without user accounts, automatic retraining, model promotion,
rollback or a gallery feedback UI. Those are separate subsequent features.

## Validation

Validation covered 408 automated tests across the project, including concurrent
judgment writes, idempotence, corrections/removals, old-generation judgments,
recording during collection refresh, empty results, INT8/FP32 image runs,
resident search, reranker/budget provenance, atomic exports and CLI behavior.
Black, Flake8, Ruff and native/Windows/Linux mypy checks passed.

A real CLI lifecycle check ran against an isolated copy of the existing
14-image synthetic document index using Tesseract-derived OCR and ONNX search.
It exercised text/image recording and set/correct/remove/re-label/export, then
checked an installed-package multilingual search and export. These were test
labels, not human relevance judgments or a measurement of retrieval quality.
The original personal indexes were not given synthetic feedback. Local test
artifacts are retained under `runs/relevance-feedback-v1/`. Native execution was
validated on macOS; cross-platform type checks do not establish native Windows
or Linux behavior.
