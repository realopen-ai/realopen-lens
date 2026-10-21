# Duplicate groups validation — 2026-10-21

This branch adds exact decoded-pixel result collapse and reviewable near-duplicate
suggestions over existing production indexes. It preserves file-byte identities,
source paths, saved judgments and virtual memberships. No model or runtime
dependency was added. See the [guide](../duplicate-groups.md) for policy and usage.

## Executed validation

- 479 Python tests passed; one optional OCR test skipped.
- 28 desktop/mobile Chromium tests passed, including the final scan-state UI changes.
- Black checked 183 files. Flake8, Ruff and mypy native/win32/linux checks passed
  for 148 source files. TypeScript, Vite and Prettier checks passed.
- Native production text and image search ran on isolated FP32 and INT8 snapshots.
  Each precision retained five before/after text query records and three image
  cases: indexed self-exclusion, inclusion, and re-encoded external self-exclusion.
  Repeated scans reused fingerprints, and the five text outputs per precision
  stayed unchanged after the final grouping refinements.

Native execution used macOS arm64, Python 3.12.11 and two inference threads.
FP32 denotes SigLIP2; E5 remains INT8. Pipeline/model/dependency metadata and source
hashes are retained with run artifacts. No new relevance accuracy score is claimed.

| Source snapshot | FP32 personal | INT8 reference |
| --- | ---: | ---: |
| Indexed file-byte identities | 2,227 | 2,052 |
| Successfully fingerprinted still images | 2,221 | 2,047 |
| Unsupported multi-frame images | 6 | 5 |
| Image errors | 0 | 0 |
| Exact pixel groups | 6 | 0 |
| Redundant exact identities | 6 | 0 |
| Near-duplicate suggestions after exact-alias consolidation | 44 | 16 |
| Suggestions with differing normalized OCR text | 43 | 15 |

The initial uncached fingerprint scans took 51.73 seconds and 27.96 seconds
respectively. A fresh final FP32 run took 20.49 seconds with the source files
already read by earlier checks; it reproduced all six exact groups, 44 suggestions
and eight text/image cases. Subsequent scans reused all eligible still-image fingerprints. These are diagnostic
observations on one host, not controlled hardware performance claims. Peak RSS and
physical 8 GB execution were not measured.

## Concrete results and failure analysis

The previously identified pizza JPEG/PNG pair was one of six groups with identical
canonical pixels and different file hashes. Before scanning, the pizza search
contained both file identities. After scanning, it retained one result with both
identities' available paths; the result set still contained 50 items. A second
exact group also appeared in that query. Other diagnostic queries covered invoices,
terminal errors, Arabic invoices and French landscapes. Individual baseline and
actual results remain available; these queries are not an independent quality set.

Low-resolution perceptual matching also surfaced screenshot/conversation templates
and generated recovery documents with changed text. Visual similarity cannot
establish equivalent content. Suggestions now carry an OCR-text-difference warning;
confirmation remains organizational metadata and never suppresses either result.
No precision/recall estimate for near-duplicate detection was inferred from these
unreviewed suggestions. Whole-frame hashing can miss crops, rotations and larger
edits; low-contrast inputs are deliberately excluded from approximate suggestions.

Regression tests exposed and covered two important boundaries:

- Self-exclusion must remove a pixel alias even when the indexed query is outside
  the selected virtual collection. Re-encoded external queries also exclude known
  matching pixels, without excluding resized near matches.
- Ranking beyond the display limit is needed for exact-copy backfill, but thumbnail
  validation must remain bounded to displayed results. Collapse and truncation now
  precede result construction for both text and image search.

Additional coverage checks orientation, transparency, animation exclusion,
interrupted/resumed scans, corrupt-cache rebuilding, stale-generation invalidation,
source removal, saved/exported feedback, source immutability, CLI/API lifecycle and
selection of every exact-copy identity when adding a combined result to a virtual
collection. Hybrid/learned candidate budgets remain unchanged; duplicate-heavy
bounded pools can return fewer unique results than requested.

## Gallery validation

Native desktop dark and mobile light sessions inspected exact groups and originals,
submitted a scan through SSE, confirmed/reset a suggestion and paged through the
50 combined groups in the FP32 snapshot. Screenshots were inspected. Successful
sessions recorded no JavaScript errors, external requests or horizontal overflow.
Review controls are disabled during a scan; index task changes refresh the panel's
snapshot state. Scan completion is followed by a new search to update displayed
search results.

## Reproduction and artifacts

```sh
uv run --no-sync python -m benchmarks.experiments.duplicate_groups \
  --source-index runs/personal-index-fp32 --collection personal \
  --output runs/duplicate-groups-fp32-new

uv run --no-sync python -m benchmarks.experiments.duplicate_groups \
  --source-index runs/retrieval-evaluation-index --collection retrieval-v2 \
  --output runs/duplicate-groups-int8-new
```

The output must be new. Original image paths and local models must remain available.
The runner copies a published index snapshot and does not copy feedback or deployed
model state. It retains scan details, individual text/image results and a summary.

Ignored local evidence is retained under `runs/duplicate-groups-v1/`: `fp32/` and
`int8/` contain initial and final scans, summaries and per-query JSONL;
`validated-fp32/` contains a fresh final-run reproduction; `gallery/` contains native
screenshots, the browser script and its summary. Earlier diagnostic evidence is
preserved separately from final scan files.

Windows/Linux native execution, constrained-memory validation, independently
reviewed approximate-match quality, crop-aware detection and automatic duplicate
handling beyond exact pixels remain future work. Existing image identities and
judgments have no destructive migration.
