# Local gallery and FastAPI adapter

The gallery uses the same indexing, search, image-similarity and feedback operations
as the CLI. The engine stays independent of FastAPI and React. Normal runtime
uses local files and compiled assets, without a Vite server, CDN, remote fonts,
telemetry or model downloads.

Search one folder or choose **All folders** for [library-wide text and image search](library-search.md),
including source labels, cross-folder exact-content grouping and separately scoped feedback.
Select results to save them in a [cross-folder virtual collection](virtual-collections.md);
text and image searches stay within that collection until you change scope.

Standalone serving uses the same [RAM-based precision default](siglip2-precision.md)
as the desktop launcher. Existing folder profiles and manual overrides take priority.

## Build and run

From a checkout, install Node.js 22.12+ and the existing Python/Tesseract
prerequisites. Download models explicitly with the existing `lens models fetch`
command if they are not already installed. Dependency installation requires
network access once; subsequent gallery use does not.

```sh
cd frontend
npm ci
npm run build
cd ..
uv sync --locked --extra web --no-editable --reinstall-package realopen-lens
uv run --no-sync lens serve --index-dir runs/personal-index-fp32 --allow-root data
```

Open **http://127.0.0.1:8765**. The existing FP32 collection is used automatically;
the server does not rebuild it or change its precision. Use another `--index-dir`
for an INT8 index or a fresh library. `--models`, `--threads` (1–8) and `--port`
configure the local engine and listener. Rebuild and reinstall the package after
changing frontend assets. Built assets are included in the wheel when present;
the core package also builds without them or Node.js.

`--allow-root` is repeatable. Imports must resolve inside one of these roots or an
existing registered collection root. With no explicit option, the current user's
home directory is allowed, including its subfolders. Operating-system permissions
still apply; Lens does not scan anything until a folder is added for indexing.
An explicitly empty API allowed-root list permits
refreshing existing collections but does not authorize new arbitrary folders.
The form's **Choose folder** button browses local directories and fills the folder
path without uploading files. Navigate into a directory and press **Select this
folder**, then **Start indexing**. Choosing/cancelling alone does not index anything.
The form suggests an unused name from the selected folder and reuses the existing
name/precision when refreshing that exact folder. Custom names remain editable.
Names already bound to another folder are rejected before a job is queued; the
form stays open so the name can be corrected.
Hidden directories are omitted from browsing; the optional text path remains usable.
Directory browsing uses the same import-root boundary, including resolved symlinks.

The web dependencies are optional. The CLI, engine and benchmark tooling remain
usable without the `web` extra. If the compiled gallery is missing, `/` explains
how to build it rather than returning a blank page.

## Gallery workflow

The image library leads with search and images. **Learning**, **Duplicates**,
**Activity** and **Workspace settings** are separate sidebar destinations. Active
tasks retain a compact floating progress strip while successful completed tasks
remain in Activity. See [everyday workspaces](workspace.md) for setup/import, build
identity and a stable launch/update workflow.


- Select a saved collection and browse thumbnails, with bounded pagination.
- Search by visual meaning, exact OCR terms, OCR meaning, or default hybrid search.
- Open a result to inspect its image, dimensions, source paths, OCR preview and issues.
- Use **Similar**, or the image button in the search bar, for image-to-image search.
- Every search result shows labeled **Relevant** and **Irrelevant** controls.
  Unsaved searches show an explanation and **Search again with feedback**, which
  explicitly repeats the original text/image query with recording enabled. Nothing
  is recorded until you enable feedback. You can also select **Save for feedback**
  before searching. Click a selected judgment again to remove it. Saved judgments
  are restored when reopening the remembered search in the same browser tab.
- **Remember query feedback** applies exact-query relevance memory to text searches.
  Relevant images move first; irrelevant images disappear with **Undo feedback**.
  **Show hidden images** lets you clear remembered negatives. Rerun to fill places
  after local changes. Disable memory before collecting baseline training searches.
  See [query memory](relevance-feedback.md#remembered-text-query-feedback).
- Add or refresh a folder, selecting its collection, precision and installed OCR
  languages. Originals are never moved. Existing collections retain their precision.

Each indexed folder has a **⋯** menu in the sidebar:

- **Refresh index** scans its saved source path using its existing precision and OCR
  languages. New/changed files are processed, unchanged analyses are reused, and
  removed paths disappear from the new index. Progress, cancellation and reconnects
  use the same live task display as initial indexing. Manual refresh remains available.
- **Automatic refresh** opts this folder into metadata checks while Lens is running.
  Choose a one-, five- or fifteen-minute interval; settled changes use the same
  indexer and live progress. See [automatic refresh](automatic-refresh.md) for
  cancellation, restart and disconnected-drive behavior.
- **Rename** changes the name in Lens without renaming the source directory.
  Its stable collection ID, feedback, virtual collections, duplicate reviews and
  model activation remain attached. Saved search records retain their original
  names as historical provenance; new searches use the current name.
- **Remove from Lens** asks for confirmation and unregisters the indexed collection.
  It never deletes or changes original files. Shared analysis caches, immutable
  artifacts, saved feedback/history and detached organization/review records remain
  on disk; this is not a secure erase or storage cleanup operation. Adding the same
  path later creates a new identity without inheriting those detached records.

Rename/removal require an idle task queue. Stale folder identities and conflicting
names are rejected rather than changing a replacement collection. A source folder
must be available to refresh; it can still be renamed or removed while disconnected.
Prepared training datasets from older versions that lack collection IDs need to be
prepared again after a rename. Newly prepared datasets resolve their source by ID.

Text and similarity queries return up to 50 results in the initial UI. The API
supports up to 100. Image uploads are bounded to 64 MiB, with actual transferred
bytes displayed during upload; the engine also enforces its existing decoded-pixel
limits. Uploads live only in controlled temporary storage and are deleted after
completion, failure or cancellation; leftover uploads are removed on startup.

## Real-time progress

**Workspace settings → Device setup** checks workspace access, model integrity and
Tesseract/languages without downloading anything. A source-access check is also
available inside **Add folder**. The section starts collapsed with its title and
description. Failed checks expand it automatically, and it can still be collapsed
manually. Reports list each prerequisite separately and identify missing language
packs. Use the installed-language suggestion explicitly
to change the indexing form; requested languages are never silently removed.
Settings changes invalidate the displayed report until checked again. Existing
images stay browsable while setup needs attention.

Indexing and searches return HTTP 202 with a task ID. The browser subscribes to
`GET /api/tasks/{id}/events` using Server-Sent Events. The stream sends a `progress`
event approximately every 500 ms while a task is active, then its terminal state.
It exposes actual indexing stage/counts, search stages, elapsed time, errors and
completed results. It does not invent an overall completion percentage or model
inference token progress. Short stages can finish between snapshots.

SSE reconnects to the latest complete task snapshot, including a terminal result.
Event IDs are revisions, not a retained per-image event log; reconnecting does not
replay every intermediate update. Closing a browser tab does not cancel a job.
An explicit Stop requests cancellation: indexing checks roughly every 500 ms;
search cancellation waits for a stage boundary, with encoder calls retaining
existing deadlines (normally 120 seconds).

One application worker serializes indexing and inference, with at most four
outstanding jobs. Extra submissions return 429. Indexing/image encoding releases
resident text encoders first; repeated text searches reuse one compatible resident
session. Read-only browsing and SSE remain responsive during model work. The API
uses a separate per-index server lock to prevent multiple gallery processes from
loading duplicate workers for the same store.

Index results retain `completed_with_errors`, failed/degraded counts and the
production catalog's detailed failure information. Inspect
`GET /api/collections/{collection}/status` or `lens status` for individual indexing
failures. Failed searches never display old images as if they matched a new query.

Operational task snapshots are kept under `api-tasks/` in the index directory,
bounded to the 50 most recent tasks. Query text and result payloads are not written
to these journals. Results are held in memory during the server lifetime; after
restart, rerun a completed search to display it again. Explicitly saved feedback
runs remain in `feedback.sqlite3`. Unfinished API tasks become `interrupted` on
restart rather than being silently replayed. Resubmitting indexing uses the
engine's durable stage cache. Graceful shutdown requests cancellation and releases
owned workers; forced termination cannot perform graceful cleanup.

## Local access boundary

The standalone launcher binds only to `127.0.0.1` with one Uvicorn worker. The API
checks loopback Host headers, rejects unexpected browser origins and requires a
random per-server HttpOnly, SameSite=Strict session cookie. The gallery obtains
that cookie through `/api/session`; API clients must do the same and retain it.
There is no permissive CORS policy. Ports use distinct cookie names.

Images are addressed by collection/image IDs, not arbitrary filesystem paths.
Only registered image sources and verified thumbnails are served. Original source
signatures and file hashes are checked, missing or changed sources are rejected,
and previews use thumbnails when a browser cannot render the original format.
At most two verified immutable manifests are cached; artifact or manifest filesystem
changes invalidate that cache. Responses use no-store, a same-origin content
security policy, no-referrer and nosniff headers. Access logs are disabled by default.

This is a trusted local application, not a multi-user or network deployment.
Do not expose the factory through an external listener or multiple API workers.

## API surface

| Operation | Endpoint |
| --- | --- |
| Session/bootstrap and allowed import roots | `GET /api/session` |
| Browse local image folders (200 per page) | `GET /api/folders` |
| Saved collections | `GET /api/collections` |
| Refresh a saved folder with live indexing progress | `POST /api/collections/refresh` |
| Configure automatic refresh for an existing folder | `POST /api/collections/auto-refresh` |
| Rename/unregister an indexed folder | `POST /api/collections/rename`, `/remove` |
| Browse a collection | `GET /api/collections/{collection}/images?offset=0&limit=60` |
| Index status and detailed issues | `GET /api/collections/{collection}/status` |
| Verified media | `GET /api/collections/{collection}/images/{image_id}/thumbnail` or `/original` |
| Start text/image/indexing task | `POST /api/tasks/search`, `/similar`, `/index` |
| Check local setup with progress | `POST /api/tasks/setup` (`siglip_precision`, `languages`, optional `source`) |
| Start uploaded-image task | `POST /api/tasks/upload?collection=personal&save_run=false` (raw image body) |
| Task snapshots | `GET /api/tasks` or `GET /api/tasks/{id}` |
| Live progress | `GET /api/tasks/{id}/events` |
| Request cancellation | `POST /api/tasks/{id}/cancel` |
| Browse hidden query results | `GET /api/feedback/hidden?query=...&collection=...` |
| Active workspace and build | `GET /api/workspace` |
| Workspace preview/import jobs | `POST /api/tasks/workspace-preview` or `/api/tasks/workspace-import` |
| Saved results and judgments | `GET /api/feedback/runs/{run_id}` |
| Set/correct or remove a judgment | `PUT` or `DELETE /api/feedback/runs/{run_id}/{image_id}` |

Media accepts an optional generation ID for a prior published snapshot belonging
to the same collection. Feedback history and JSONL export remain available through
the [feedback CLI/Python API](relevance-feedback.md). No model is trained or
promoted automatically.

## Development and validation

For Vite development, start the backend with
`--dev-origin http://127.0.0.1:5173`, then run `npm run dev` in `frontend/` and open
that exact address. The development proxy forwards `/api` to port 8765. A production
wheel serves only the compiled frontend; Vite is development tooling.

```sh
uv run --no-sync pytest -q
cd frontend
npm run format:check
npm run build
npx playwright install chromium
npm test
```

CI runs the Python checks with the web extra on macOS, Windows and Linux, and runs
TypeScript/build and Chromium desktop/mobile tests on Linux. Browser contract tests
use controlled API responses; Python integration tests exercise the real adapter
and collection/feedback storage with bounded model doubles. Separate local smoke
checks exercised real Tesseract/ONNX indexing, SSE, FP32 retrieval and image uploads.

The gallery CI job reuses npm's download cache and prefers cached npm packages.
It caches Playwright browser files by exact Playwright version, runner OS/image
and architecture, and downloads only Chromium's headless shell. Cache hits skip
browser downloading. Linux system dependencies are installed on every fresh runner;
their installation has its own step so its cost is visible separately from browser
downloads. Desktop and mobile coverage is unchanged. Each run retains JSON test
timings and failure traces as `gallery-results-*` artifacts for seven days.
The first run or a Playwright version/runner image change needs a new browser cache.

Native Windows/Linux model inference remains release
work. Pixel-identical files in different encodings can still appear separately;
decoded-pixel deduplication is tracked as deferred work. Virtual collections,
automatic learning, a native folder picker and an OS application bundle are not
part of this initial gallery.

Implementation references: [FastAPI streaming responses](https://fastapi.tiangolo.com/advanced/custom-response/),
[Vite production builds](https://vite.dev/guide/build/), and
[Hatch build artifact inclusion](https://hatch.pypa.io/latest/config/build/).

### Initial validation evidence

The initial implementation passed 419 Python tests (one optional OCR experiment
skipped), 14 Chromium desktop/mobile tests, formatting/lint checks, and mypy for
native, Windows and Linux targets. The browser tests cover browsing/details,
SSE completion and partial failures, similarity requests, explicit feedback and
reload restoration, and preventing stale results after errors or restarts.

A separate native check indexed three fresh synthetic documents through the
production Tesseract/ONNX pipeline: all three were searchable, with zero failed
or degraded files. The API returned before indexing finished, and seven SSE
snapshots included live OCR, vision and semantic stages. Search recording and
feedback set/remove were also exercised on that isolated test collection.

The installed gallery was checked against an existing 2,227-image FP32 collection:
text, indexed-image similarity and a newly encoded uploaded image each returned
50 results. Desktop/mobile screenshots were inspected; the check recorded no
JavaScript errors, horizontal overflow or external requests. These are integration
checks, not new retrieval-quality or constrained-hardware benchmarks.

That real-upload check found a metadata-only ctime change incorrectly rejected as
changed query content. The image-query guard now rehashes that narrow case while
still rejecting changed bytes, modification times or file identities. Regression
tests cover both unchanged bytes and altered bytes with restored size/mtime.
Core indexing identities and the deferred decoded-pixel deduplication are unchanged.

Local results, SSE snapshots, diagnostics, scripts and screenshots are retained
under `runs/local-gallery-v1/` and remain ignored because they can contain private
paths or collection images. Both core-only and gallery-containing wheels were
built successfully; installed Python and compiled-asset hashes matched checkout.

## Reviewed local learning

The **Learning & models** panel connects explicit feedback to reviewed dataset preparation,
logistic training, production evaluation and explicit activation/rollback. Jobs use the same
live progress and cancellation controls. See [feedback training](feedback-training.md) for
review requirements, compatibility and metric limitations.

## Appearance

Use the Light/Dark switch in the header to choose an appearance. Lens follows the
system preference until you select a mode, then remembers your choice locally in
the browser. The preference is applied before the gallery loads. If browser
storage is unavailable, the switch still works for the current page session.

## Virtual collections

Create named selections from card checkboxes or the sidebar, then browse and search
their members. Rename/delete controls and revision checks keep concurrent edits
explicit. See [virtual collections](virtual-collections.md) for membership semantics,
source refreshes, CLI commands and scoped-feedback limitations.

## Duplicate review

The **Duplicate groups** panel scans the selected source with live SSE progress,
then shows exact pixel groups and reviewable near-duplicate suggestions. Scanning
enables exact-copy collapse in subsequent searches; folder browsing retains file
identities. See [duplicate groups](duplicate-groups.md) for refresh and review rules.


Indexing progress distinguishes [processed and cached stage work](stage-cache.md),
including an OCR summary at completion. A refresh does not imply rerunning OCR on
every indexed image.
