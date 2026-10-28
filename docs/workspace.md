# Everyday local workspace

A workspace keeps the collection catalog, derived indexes, recorded search/feedback
history, virtual collections and duplicate reviews in one directory. Original images
and model weights stay at their existing paths. Workspace setup does not download,
reindex, train, activate a model, or modify the source indexes.

The default directory is persistent application data:

| Platform | Default |
| --- | --- |
| macOS | `~/Library/Application Support/RealOpenLens` |
| Windows | `%LOCALAPPDATA%/RealOpenLens` |
| Linux | `$XDG_DATA_HOME/realopen-lens`, or `~/.local/share/realopen-lens` |

`lens serve` uses this directory when `--index-dir` is omitted. An explicit
`--index-dir` continues to work. Imported workspace metadata remembers the absolute
model directory; `--models` overrides it. Original source-image roots must remain
available. Moving images or model files requires an explicit path/update workflow.

## Preview and import

The importer currently combines **one matching source collection** across up to 16
indexes. Their collection ID, image root, active generation and manifest checksum
must match. Unrelated or differently indexed collections are refused rather than
merged into a misleading catalog.

```sh
uv run --no-sync lens workspace info

uv run --no-sync lens workspace import \
  --source runs/virtual-collections-v1/validated-fp32/index \
  --source runs/duplicate-groups-v1/validated-fp32/index \
  --log
```

This previews only. Review the destination, counts, exclusions and `plan_sha256`.
Stop other sessions editing the sources while importing. To apply that exact plan:

```sh
uv run --no-sync lens workspace import \
  --source runs/virtual-collections-v1/validated-fp32/index \
  --source runs/duplicate-groups-v1/validated-fp32/index \
  --apply PLAN_SHA256 --log
```

Use `--index-dir NEW_DIRECTORY` for another destination and `--models DIRECTORY`
when model weights are stored elsewhere. A stale plan is rejected. Existing saved
data is never overwritten. An uninitialized destination containing only server/task
journals can be initialized while preserving those journals. Other destination
servers/indexers must be stopped; their locks are respected. A server can initialize
its own empty workspace through its serialized task queue.

Successful imports copy verified derived artifacts and publish the catalog only
when the complete workspace is ready. Exceptions/cancellation before publication
remove staging files. Importing a large collection reports live file-checking/copying
progress; keyboard interruption cancels CLI preparation. Interrupted imports are
restarted with a fresh preview, not resumed from an unverified partial destination.

## What is preserved

- Saved run IDs, original payloads and ranking/source provenance.
- Explicit feedback revisions, timestamps and clear actions. Shared event histories
  are deduplicated and each source's ordering is retained. Global event IDs are
  reassigned in the destination; the source databases remain untouched.
- Latest per-run labels and effective exact-query memory reconstructed from events.
- Virtual collection IDs, revisions and membership, including unavailable members.
- Compatible duplicate snapshots, fingerprint caches and confirm/dismiss reviews.
- Available analysis caches and complete historical generation artifacts.

Independent contradictory query judgments—including exact pixel aliases—are refused.
An earlier shared history prefix can be extended by a recorded later correction.
Conflicting saved payloads, event revisions, virtual collection revisions/names,
duplicate snapshots or review decisions also require choosing an authoritative
source. There is no automatic last-writer-wins policy across independent databases.

Model weights, trained reranker artifacts/activation and old API task history are
explicitly excluded. They remain in the source directories. The new workspace uses
ordinary hybrid ranking until a compatible model is explicitly reviewed/activated.
This importer is an initialization tool; repeated synchronization into an existing
workspace and unrelated multi-source collections are not implemented.

## Gallery and long tasks

The library keeps search and images together. **Learning**, **Duplicates**,
**Activity** and **Workspace settings** have sidebar destinations. Tool forms retain
state when switching destinations. Query-memory explanations and hidden images are
inside the compact query-feedback disclosure. Feedback buttons and undo remain
available in the library. Completed successful tasks move out of the image area;
active work and errors retain a floating progress strip with cancellation.

**Workspace settings** shows the exact active index path, remembered model path,
application version, code/gallery fingerprints and installation path. When started
from a checkout, it reports whether the installed code/assets match that checkout.
This is a diagnostic comparison of local files, not a signed software verification.
Restart after changing/installing code so the running process loads a coherent build.

The same page can preview/import directories. Both operations use the bounded task
queue and reconnectable SSE progress. Preview must complete before **Create workspace
from this preview** is enabled. Input edits invalidate it. Importing into the current
empty workspace refreshes the library; importing elsewhere leaves the current
session on its original index and displays the new launch command.

## Build, update and launch

First-time dependencies and models follow the [gallery setup](local-gallery.md#build-and-run)
and [collection setup](collections.md#setup-and-first-collection) guides. Stop the
old server before updating its installed build. For an existing environment:

```sh
npm --prefix frontend run build
uv pip install --no-deps --reinstall-package realopen-lens .
uv run --no-sync lens workspace info
uv run --no-sync lens serve
```

This refreshes the application without changing installed runtime dependencies.
When dependency definitions change, use the documented locked environment sync
with the `web` extra and `--no-editable` instead. Editable installs rely on Python
`.pth` processing; environments marking those files hidden may skip them. The
regular installation above avoids that dependency. Rebuild/reinstall after changes;
`--no-sync` deliberately does not refresh installed code.

Normal access remains **http://127.0.0.1:8765**. Test servers may use isolated ports;
normal updates do not require switching ports or workspace directories.
