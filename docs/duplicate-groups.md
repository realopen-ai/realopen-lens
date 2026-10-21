# Duplicate groups

Duplicate scans organize existing indexed images without moving or deleting files,
changing file-byte identities, or rerunning OCR and model inference.

## Gallery

Open **Duplicate groups** and choose **Scan duplicates**. The scan covers the
selected source folder, including when a virtual collection is open. Progress and
cancellation use the existing task/SSE controls. Interrupted scans retain completed
fingerprints; submit the scan again to resume. Only a completed scan publishes a
new grouping snapshot. Per-image errors and unsupported counts are reported.

The panel shows two kinds of groups:

- **Identical pixels:** distinct file-byte identities with exactly matching
  canonical decoded pixels. New text and image searches display one representative
  with all available source paths. Search again after scanning to update results.
- **Possible near-duplicates:** visually similar whole-frame pairs, with their
  exact-copy aliases included. Inspect the originals, then confirm, dismiss or
  reset the suggestion. Reviews persist locally. Confirmation does not hide images
  in search, merge identities, propagate relevance labels or delete files.
  Confirmed links constrain split isolation when preparing a new training dataset;
  see [feedback training](feedback-training.md).

OCR differences produce an explicit warning. Similar layouts can contain different
text; matching fingerprints at reduced resolution do not prove equivalent content.
Groups are independent pair suggestions, not transitive clusters: an A/B match and
B/C match do not imply an A/C match.

Folder and virtual-collection browsing retain individual file-byte identities for
organization and stable pagination. Selecting a combined search result selects all
its available exact-copy identities, within the existing 500-ID action limit.
Virtual searches combine only members of that virtual collection. Membership itself
is never rewritten or expanded automatically.

## CLI

From an installed checkout:

```sh
uv run --no-sync lens duplicates scan \
  --index-dir runs/personal-index-fp32 --log

uv run --no-sync lens duplicates list \
  --index-dir runs/personal-index-fp32 --limit 20

uv run --no-sync lens duplicates review PAIR_ID confirmed \
  --index-dir runs/personal-index-fp32
```

Review states are `confirmed`, `dismissed` and `unreviewed`. All commands accept
`--collection`, defaulting to `personal`. List supports `--offset` and `--limit`.
The scan emits structured JSON on stdout; optional progress goes to stderr. It
exits nonzero on image errors. Unsupported multi-frame inputs are counted separately.

## Identity and search behavior

Exact fingerprints use EXIF-oriented RGBA pixels, dimensions and a versioned SHA256
header. RGB under fully transparent pixels is zeroed; partial alpha is preserved.
Color-profile metadata is not color-managed. Multi-frame images are excluded so
matching first frames cannot hide different animation/page content. Decoding is
bounded to 40 megapixels per image.

Near suggestions use a 256-bit horizontal difference hash, 16×16 RGB appearance
on white, aspect ratio and contrast. The current bounds are Hamming distance ≤8,
RGB mean absolute difference ≤12, relative aspect change ≤2%, and contrast ≥8.
A BK tree avoids allocating a quadratic distance matrix. Comparisons are capped
at five million and output at 10,000 suggestions; exceeding a limit fails the scan
without replacing the previous published snapshot. Crops, rotations and larger
edits can be missed. These thresholds are heuristic, not a calibrated probability.

Exact collapse keeps the first ranked member and its original ID, metadata and
score, appending the other available paths and `duplicate_ids`. Visual, semantic
and lexical searches backfill before constructing displayed results. Hybrid and
learned search retain their existing bounded candidate pools; duplicate-heavy pools
can therefore return fewer unique results than the requested limit. This branch
does not change candidate scores or silently widen learned-model candidate budgets.

Image-query self-exclusion removes all known pixel-identical aliases, including
re-encoded external queries eligible under the fingerprint policy. Near matches
remain candidates. `--include-self` includes one result per exact group.

Saved feedback remains bound to its original file IDs and displayed representative.
New saved searches record grouping policy/snapshot provenance; exports include exact
alias IDs when present. Existing judgments are not migrated or relabeled. Review
source/duplicate groups across training splits, and do not mix different ranking
source versions in a prepared dataset.

## Storage, refresh and API

The index's `duplicates/` directory stores versioned per-content fingerprint caches,
atomic grouping snapshots and source-scoped review state. Source bytes are checked
before and after fingerprinting; publication verifies source signatures and active
generation. Invalid cache entries are recomputed. The indexing writer lock prevents
concurrent scans/indexing from publishing inconsistent state.

Snapshots bind the source manifest's image metadata/signatures and pipeline. After
an index refresh or pipeline change, scan again; the gallery reports when the
current snapshot has no scan. Compatible fingerprints are reused. Old snapshots
remain available as historical artifacts. Source files, catalog generations,
feedback records and virtual memberships retain their existing identities.

The authenticated local API adds:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/tasks/duplicates` | Queue a scan with `{ "collection": "personal" }` |
| GET | `/api/duplicates` | Browse groups with collection, offset and limit |
| POST | `/api/duplicates/{id}/review` | Store collection and review state |

Scans use the existing bounded task queue, SSE stream and cancel endpoint. No new
runtime dependencies or network services are required.

See the [validation report](benchmarks/2026-10-21-duplicate-groups.md).
