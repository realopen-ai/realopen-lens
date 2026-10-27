# Search pagination

Gallery text searches and image-similarity searches show 50 results initially.
Use **Load more results** below the grid to continue. Folder, library-wide and
virtual-collection scopes remain attached to the search. This is separate from
the date-ordered image library's existing browsing pagination.

## Ranking and limits

The ordinary first page is preserved, including the active reranker, metadata
interpretation, exact-query feedback and exact-image grouping. A second retrieval
builds a deeper pool once, expanding hybrid candidates to 1,000 per channel and
using hybrid fusion without the trained reranker. Continuations append previously
unseen groups after the original head; they never reorder that head. Similarity
continuations use visual similarity. The gallery identifies hybrid continuation
when the first page used a learned ranker.

The combined pool contains at most **1,000 image groups**. The displayed count is
the retrieved pool size, not the number of relevant images or a complete library
count. Later results can be weak matches. A cap message recommends refining the
query. Lexical and metadata constraints can produce fewer results.

Preparing the initial pool costs additional retrieval, validation and memory.
Later pages slice the retained ranking without model inference. Snapshots are
local and ephemeral: at most eight are retained, for 30 minutes from creation.
Restarting the server or eviction requires a new search. An index generation,
folder identity, virtual membership/revision, duplicate snapshot or pinned source
signature change rejects continuation with a retry message rather than mixing
snapshots. Some unrelated duplicate scans can conservatively invalidate cursors.

New query-memory negatives are checked when serving pages. New positive decisions
are shown with their current label but do not reorder the frozen page sequence;
repeat the search to apply updated promotion throughout the ranking. Removing a
negative does not restore images omitted when the pool was constructed until a
new search. Hidden images can leave a short or empty page with another cursor.

## Feedback and evaluation

When recording is enabled, all pages share one saved run. Only explicit judgments
become labels; unloaded images are not implicit negatives. First-page candidate
features keep their original provenance. Deeper labels support query memory and
raw export; they have no original-budget training features. Dataset preparation
rejects runs containing those featureless labels with an explicit requirement for
a separate deeper-candidate evaluation. No model is trained or promoted here.

Pagination is opt-in at the API (`paginate: true`) and enabled by the gallery.
Ordinary API/CLI queries and benchmark calls retain their existing defaults and
candidate budgets. Recall@50, MRR@10 and nDCG@10 evaluations have not been
redefined as pool coverage. No relevance improvement is claimed for retrieving
more images.

## API

Start a normal SSE-backed task with `paginate: true` on `/api/tasks/search` or
`/api/tasks/similar`; image uploads use `/api/tasks/upload?paginate=true`.
`limit` remains the page size (1–100). Completed results include `next_cursor`,
`total`, `pool_limit`, `truncated` and the pagination policy.

Fetch `/api/search-pages?cursor=...` using the returned opaque cursor. Continuations
carry the same run ID and absolute ranks. The cursor identifies its original query
and scope; callers do not resubmit a query, mode or folder. Expired or stale
snapshots return HTTP 409. Existing local session authentication still applies.

## Reproduce the native diagnostic

```sh
uv run --no-sync python -m benchmarks.experiments.search_pagination \
  --index-dir /path/to/index --models models --collection personal \
  --output runs/search-pagination-new
```

This compares ordinary and paginated heads, traverses every page, checks image and
duplicate-group uniqueness, retains individual results and timings, and samples
process-tree RSS. It uses three text queries and one stored-image query through
both folder and library APIs. It supplies no relevance ground truth.

See the [native observations](benchmarks/2026-10-26-search-pagination.md).
