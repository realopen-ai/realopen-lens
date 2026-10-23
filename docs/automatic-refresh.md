# Automatic folder refresh

Open a folder's **⋯** menu and enable **Automatic refresh**. Choose checks every
minute (the default), five minutes or fifteen minutes. It is off by default for
all folders. Each folder keeps its own preference, attached to its stable ID;
renaming the folder in Lens preserves it. Removing the folder clears the preference.

Lens checks while the gallery server or desktop launcher is running. It does not
install a background OS service. Close Lens to stop monitoring; saved preferences
resume on the next launch. The next check compares the folder with the published
index, so additions, changes and removals made while Lens was closed are detected.

## What happens when files change

Checks read supported image paths and filesystem metadata, without decoding images,
hashing their contents or loading models. The monitor excludes `.git`, directory
symlinks and the index directory. A changed folder must produce the same snapshot
on another check at least ten seconds later before a refresh is queued. More
changes restart that settling period, batching ordinary bulk copies.

Automatic refresh uses the same incremental indexer as **Refresh index**, preserving
the collection's SigLIP2 precision and OCR language settings. Unchanged content and
completed processing stages are reused. There is one inference worker; monitoring
only queues work when the task queue is idle, and queues at most one refresh per
pass. Searches submitted after indexing starts wait for that task or its cancellation.

The gallery discovers background jobs without reloading, then connects to their
normal SSE progress stream. The floating progress strip identifies **Refreshing
automatically**; Activity retains the task outcome. Browsing the affected folder
updates after successful indexing when no image dialog, selection or other tool
is active. Otherwise a notice invites you to select the folder again; ongoing
work and dialogs stay open. Existing search results are not reranked silently.

**Stop** on an automatic job cancels it and switches automatic refresh off for that folder so it
does not immediately restart. Enable it again to resume. A failed automatic job
also pauses the folder with an error message. `completed_with_errors` retains
normal per-image failure/degradation reporting and does not disable future changes.
Turning off the checkbox stops future scheduling; use **Stop** to cancel work that
has already been queued. Closing the server interrupts owned work but retains the
preference for next launch, allowing the indexer's cached stages to be reused.

## Unavailable folders and safeguards

A missing folder or unreadable subtree produces a visible check error and is retried
at its configured interval. Detection failures never schedule an index. Restoring
the same directory allows monitoring to resume. If the path points to a different
directory or an empty mount point after a drive disconnects, its device/inode
identity differs: automatic refresh waits for review. Restore the original directory
or disable/re-enable the option after verifying the new source. Automatic jobs also
check root identity before processing and immediately before publication.

Metadata checks are bounded to 100,000 filesystem entries and ten seconds of
cooperative scanning per folder. A limit error recommends manual refresh, which
remains available for larger trees. Stop/time limits are checked between filesystem
operations; an individual blocked OS/filesystem call cannot be forcibly interrupted
by this metadata thread. Slow or network-mounted filesystems may therefore take
longer to respond. All this applies to the registered root; it is not a general
volume-management or filesystem-event service.

Same size, nanosecond mtime/ctime, device and inode are treated as unchanged, matching
normal incremental indexing. Metadata-preserving edits are not a content-integrity
scan; use the CLI's `index --rehash` when full byte verification is required.

Settings are stored atomically in `INDEX_DIR/folder-refresh.json`, separately from
catalog schema and desktop preferences. They contain folder IDs, intervals and root
identity, not image contents. Malformed settings disable automation without blocking
the gallery. Explicitly re-enabling preserves a bounded invalid file as
`folder-refresh.invalid-<id>.json` before replacing it.

## API

Authenticated local clients configure an existing folder using its current name
and stable ID:

```json
POST /api/collections/auto-refresh
{
  "collection": "personal",
  "collection_id": "<current folder ID>",
  "enabled": true,
  "interval_seconds": 60
}
```

Intervals of 30–3600 seconds are accepted by the API. `GET /api/collections` includes
an `auto_refresh` object per folder with its enabled flag, interval, check state,
error, last-check timestamp and current task ID. Tasks expose `origin: "automatic"`
and the stable `collection_id`; task listing, SSE progress and cancellation use
existing endpoints. The gallery uses `GET /api/tasks?summary=true` for lightweight
discovery, fetching a completed result once when needed. Automatic refresh never authorizes a new import root or changes
source files.
