# Timeline gallery validation — 2026-10-25

## Behavior

Browsing now orders images by recorded calendar timestamp and groups them by month
and day. Capture metadata takes priority over the indexed file-modification
fallback; missing dates stay last in both directions. Library browsing includes
all folders and retains existing decoded-image deduplication and source references.
Search ranking is unchanged.

Pages include order, generation and timeline snapshot provenance. Tests cover
stable ties, timezone-free capture dates, invalid dates, unavailable copies,
revision/index changes, scope isolation and page coverage. Library browsing does
not initialize inference models or construct an OCR lexical index.

## Existing-index check

Read-only production browse calls covered an existing 2,227-entry FP32 index and
a small mixed-precision metadata control index. Each page contained at most 60
results. Every available image/group appeared exactly once, ranks continued across
pages, and all known dates were in descending calendar order.

| Scope | Images/groups | Pages | Median page call | Slowest page call | Highest post-page process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing folder | 2,227 | 38 | 0.440 s | 0.817 s | 77.6 MiB |
| Existing library, decoded copies grouped | 2,221 | 38 | 0.526 s | 3.012 s | 270.2 MiB |
| Metadata control folder | 6 | 1 | 0.008 s | 0.008 s | 181.6 MiB |
| Mixed-precision control library | 6 | 1 | 0.006 s | 0.006 s | 181.7 MiB |

The older large index has no extracted capture dates and therefore exercises the
modification fallback for every image. Each control scope contains five captured
images and one modification fallback. Separate regression fixtures exercise real
EXIF capture metadata and intentionally unknown dates.

These are sequential local macOS observations, not a controlled before/after
performance comparison. The same process handled the scopes in table order;
allocator/cache retention affects RSS. Memory was sampled after each page, not
continuously, so these are not peak-memory measurements. Source files, index
contents and model weights were not modified by this check. Raw local artifacts
are retained under `runs/timeline-gallery-v1/`.

## Automated and visual checks

831 Python tests passed with one optional skip; 102 desktop/mobile browser tests
passed. Black, Flake8, Ruff, mypy (native, Windows and Linux targets), TypeScript,
Prettier and the production gallery build passed. The rebuilt macOS archive passed
12 frozen-package smoke checks and its SHA-256 check. A separate extracted-app
HTTP check covered folder, library and virtual chronology and paged ranks against
an isolated copy of the metadata control index. Browser
coverage includes scroll loading, continuous headings, selection retention, sort
changes, folder/library/virtual scope, stale-page reload, navigation across a page
boundary, background-refresh scope/order and unchanged search presentation.
Light-theme desktop/mobile layouts were inspected for grouping and horizontal
fit. Existing theme regressions remain part of the full browser suite.

## Limits

Capture timestamps without offsets cannot establish a global instant; the timeline
keeps their recorded calendar values. File modification dates can reflect copying
or editing. Loaded cards remain in the DOM; response pagination is bounded, but
this is not a virtualized gallery. A date-jump rail and continuous peak-memory
profiling are not implemented. Native Windows/Linux execution was not run locally.
