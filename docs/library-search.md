# Search across folders

Choose **Image library**, or select **All folders** under **Search in**, then enter
a text query. This searches the indexed folders in the current workspace. Results include the source folder names. Open an image to see all
available source paths; originals remain untouched.

Selecting a folder restricts browsing and search to that folder. Virtual
collections, image similarity and bulk organization remain folder-scoped. From a
library result, **Similar** searches the representative image's source folder.

## CLI and Python

```bash
lens search "pizza" --all-folders --index-dir /path/to/workspace --limit 50
lens search "invoice number" --all-folders --index-dir /path/to/workspace --mode lexical
lens search "pizza" --all-folders --index-dir /path/to/workspace --interactive
```

Library CLI output is an object containing `results`, `ranking`, feedback state
and availability diagnostics. Existing folder search output remains unchanged.

```python
from pathlib import Path
from realopen_lens.retrieval.library import LibrarySession

session = LibrarySession(index_dir=Path("/path/to/workspace"), models=Path("models"))
try:
    result = session.search("pizza", limit=50)
finally:
    session.close()
```

## Ranking and resource use

Library search captures published folder generations, validates current source
paths, and computes global retrieval channels. Folder-local ranks are not used as
votes: splitting a folder does not give its images extra ranking weight.

The existing hybrid policy combines visual cosine similarity with bounded OCR
evidence. Lexical BM25 and term frequencies are computed over the combined unique
content, and semantic passage scores are reduced to one score per image before
fusion. Folder-specific experimental rerankers are not applied to library results.

INT8 and FP32 folders use their matching SigLIP2 text encoders. Query vectors are
reused across folders with the same precision; E5 remains INT8 and is encoded once
per query. Supported folders must match the pinned encoder/preprocessing and
runtime dependency versions. An incompatible or corrupt published folder fails
the request explicitly, rather than silently omitting it.

Mixed-precision visual/hybrid results are marked **mixed precision (experimental)**
in the existing ranking caption and API ranking metadata. Semantic-only and lexical
search share a single score space and do not receive this marker.

Mixed-precision visual scores use their original cosine scale. INT8 quantization
can change relative rankings; this policy is measured diagnostically against the
existing judgments and does not claim calibrated probabilities or independent
relevance validation. [The diagnostic comparison](benchmarks/2026-10-23-library-search.md)
records the measured tradeoffs and rejected normalization/fusion experiments.

Only one query worker is resident at a time. Switching precision unloads its
models before loading the other precision. Verified file signatures are retained
for reuse and checked again before encoding; changed model files trigger normal
integrity verification. Mixed libraries therefore pay model-switching costs.

Embedding matrices are memory-mapped and processed one folder at a time. Combined
metadata, OCR text, scores and an in-memory lexical index scale with library size.
The initial implementation limits searches to 256 folders and 100,000 indexed
image entries. It does not create a second permanent vector index.

## Duplicates and feedback

Identical file bytes collapse across folders automatically. Published duplicate
scans also allow identical decoded pixels across formats to collapse. Without
those scans, cross-format copies may remain separate. Approximate matches are not
hidden automatically. If the same content has both precisions, a deterministic
FP32 representative supplies its scores; other locations remain attached.

**Save for feedback** records the library search and its source generations.
Relevant/irrelevant judgments apply to that exact query across the library. They
are separate from folder and virtual-collection judgments. **Show hidden images**
lets you review and clear library judgments without model inference.

Library feedback can be exported with its provenance. Existing reranker training
rejects these runs: mixing multiple source configurations requires a separate
library-aware training/evaluation protocol. No model is trained or activated by
marking a result.

## Progress and availability

Search uses the normal background task queue, live SSE progress and **Stop**
control. Cancellation is cooperative at pipeline checkpoints; an in-flight model
request is bounded by the existing encoder timeout.

Changed or unavailable source files are excluded and reported. Folders without a
published generation are listed as pending. A search uses its captured immutable
generations; subsequent refreshes take effect on the next search. Renaming or
removing folders does not rewrite historical feedback.

For API clients, submit `all_folders: true` to `POST /api/tasks/search`.
Folder-specific virtual scope cannot be combined with this flag. Results include
`sources` with collection IDs, generation IDs, original image IDs and paths, so
media requests can use the stable source collection ID and generation through
`GET /api/sources/{collection_id}/images/{image_id}/{kind}`. These URLs continue
to work after folder renames and stop resolving after unregistration.
Hidden judgments are available through `GET /api/library/hidden` with `query`,
`offset` and `limit` parameters.

Workspace import currently accepts single-folder sources and their folder-scoped
feedback. Importing library-scoped feedback is not supported; it is refused rather
than silently merged into another library.
