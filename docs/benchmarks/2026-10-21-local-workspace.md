# Local workspace and gallery validation — 2026-10-21

Workspace initialization and the image-first gallery were validated on compatible
published FP32 indexes. No model was trained or activated and no source index was
modified. Raw histories, local paths and screenshots remain in ignored artifacts.

## Import evidence

Two compatible source indexes contributed one 2,227-image collection, 28 saved
searches, 161 unique feedback events, 145 current per-search labels, three virtual
collections and 44 duplicate reviews. Initialization preserved every logical saved
payload and event, including explicit clears. Source snapshot fingerprints were
unchanged after import. Global event IDs are reassigned while preserving each
source's ordering; saved run IDs and event revisions/timestamps remain intact.

The isolated import completed in 11.13 seconds on macOS arm64. This is a single
setup observation, not a throughput or hardware-budget benchmark. The imported
workspace exposes 50 duplicate groups (six exact and 44 approximate), and a native
resident text query returned the identical 50 IDs as the duplicate-aware source
with query memory disabled. Effective query memory was also readable after import.

An initial comparison against a source without a duplicate scan differed because
importing the reviewed pixel snapshot correctly enabled exact-copy collapse. The
comparison was corrected to hold duplicate policy constant; original failed-run
artifacts were retained. No ranking algorithm was changed to force parity.

## UI and workflow

Learning, duplicate review, activity and workspace setup now have sidebar
destinations. The library starts with search and compact result controls. Query
memory details expand on request; completed successful tasks remain in Activity.
Active jobs retain floating live progress and cancellation when navigating away.
Workspace preview/import use the existing serialized queue and SSE stream. A real
HTTP initialization into an active empty workspace emitted six preview frames and
15 import frames, then exposed all 2,227 library image IDs without a server restart.

Real native desktop/light and mobile/dark browser checks found no JavaScript errors,
external requests or horizontal overflow. The final first image appeared at 372 px
on a 1,440 × 1,000 viewport and 640 px on a 390 × 844 viewport containing existing
virtual collections. Native screenshots and input/transaction evidence are retained
under `runs/local-workspace-v1/`.

## Checks and limitations

- 530 Python tests passed; one optional OCR test skipped.
- 36 desktop/mobile Chromium tests passed, including preview-before-apply, stale
  installation detection, persistent tool destinations and background progress.
- Black, Flake8, Ruff, mypy native/Windows/Linux targets, TypeScript/Vite, Prettier
  and Git whitespace checks passed.
- Regression tests cover stale previews, shared history extensions, independent
  conflicting labels including retired pixel aliases, virtual/review conflicts,
  source-generation changes, cancellation cleanup, destination locks and initializing
  an active empty workspace without restarting its server.

Imports initialize new or operational-only empty workspaces. They do not synchronize
existing saved data or combine unrelated/differently indexed collections. Trained
model activation and model weights are excluded explicitly. Original image/model
paths must remain available. Source folders are not moved. Native execution was on
macOS only; Windows/Linux type targets do not establish native inference support or
8 GB hardware validation.
