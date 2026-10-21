# Virtual collections

Virtual collections are named selections of indexed images. Each belongs to one
source folder collection, such as `personal`. They store image identities without
moving originals or rebuilding OCR, embeddings or the source index.

## Gallery

1. Choose a source under **Folders**.
2. Use **Create virtual collection** beside **Virtual collections**, or select
   images with their card checkboxes and choose **New collection**.
3. Select a target in **Add to virtual collection**, then **Add selected**.
4. Open the collection in the sidebar to browse or search its members.
5. Select members and choose **Remove from collection** to remove their membership.
   **Manage virtual collection** provides rename and confirmed deletion controls.

Selection works on both browsing and search results. You can select loaded cards
across pages, with a maximum of 500 images per action. Creating a collection from
an active selection keeps that selection available for the add action.

The scope banner identifies the selected collection. Text searches in all four
modes, indexed-image similarity and uploaded-image searches use that scope. Use
**Search all images** or choose the source folder to return to the full library.
Dark/light themes and mobile layouts include the collection controls.

## Search semantics

Filtering happens before candidate retrieval and ranking. A member outside the
full-library top 50 can therefore be found within its collection. Image queries
can use an indexed reference outside the subset; only the result candidates are
restricted. Self-exclusion still applies.

Scoped searches use the existing hybrid/visual/lexical/semantic policies. Managed
learned rerankers apply to full-library searches; explicit reranker overrides are
rejected for scoped queries until scope-aware model evaluation is implemented.
Current learned models were evaluated on a different candidate distribution.

Each operation binds the group to its source collection ID. A group from another
source cannot be used accidentally with incompatible embeddings or metadata.
There is no cross-source aggregation, nesting or automatic grouping in this version.

## Refreshes and unavailable images

Membership uses the index's file-byte SHA256 identity. Byte-identical aliases share
a membership; re-encoded or visually similar files remain distinct identities.

Renaming or moving a source file preserves membership after an index refresh when
its bytes remain the same and the new path is indexed in the same source collection.
Missing, changed or no-longer-indexed images remain stored as unavailable members.
They are excluded from browsing/search results, and browsing reports their count.
Restoring the same content and refreshing the source index makes them available again.

Group changes do not refresh the physical index. Use the existing indexing command
or gallery folder refresh when source files change.

## Revisions and feedback

Names and memberships have a monotonically increasing revision. Duplicate adds and
removals of nonmembers are idempotent. Stale mutations return HTTP 409; CLI operations
report an error and exit nonzero. Refresh the list before retrying. Paged browsing
can bind both group revision and source generation, and search validates the group
revision at completion.

Saved scoped searches retain the group ID, name, source ID, revision, member count
and membership hash. Explicit relevance judgments remain attached to their saved
search, even if the group is subsequently changed or deleted. Exports retain the
scope metadata.

The current feedback-training preparation accepts full-library searches only.
Scoped feedback is rejected with an explanation rather than silently evaluated as
a full-library query. Continue collecting training judgments in the full library.

## CLI

Use the [gallery setup](local-gallery.md) to install the current package. Commands
below operate on the FP32 index used in the examples; use your desired index path.

```sh
uv run --no-sync lens groups create "Invoices" \
  --index-dir runs/personal-index-fp32

uv run --no-sync lens groups list --index-dir runs/personal-index-fp32

uv run --no-sync lens groups add GROUP_ID IMAGE_SHA256 \
  --revision 1 --index-dir runs/personal-index-fp32

uv run --no-sync lens groups images GROUP_ID \
  --index-dir runs/personal-index-fp32

uv run --no-sync lens search "invoice" --virtual-collection GROUP_ID \
  --index-dir runs/personal-index-fp32

uv run --no-sync lens similar --image-id IMAGE_SHA256 \
  --virtual-collection GROUP_ID --index-dir runs/personal-index-fp32
```

Replace the IDs with those returned by the group and search commands. Mutation
commands require the current revision from `groups list` or the preceding response:

```sh
uv run --no-sync lens groups rename GROUP_ID "Receipts" \
  --revision CURRENT_REVISION --index-dir runs/personal-index-fp32

uv run --no-sync lens groups remove GROUP_ID IMAGE_SHA256 \
  --revision CURRENT_REVISION --index-dir runs/personal-index-fp32

uv run --no-sync lens groups delete GROUP_ID \
  --revision CURRENT_REVISION --index-dir runs/personal-index-fp32
```

Use `--collection SOURCE_NAME` for a source other than `personal`. Search/similarity
also accept `--virtual-revision` to require a specific membership/name revision.
`groups images` supports `--offset` and `--limit` for browsing.

## Storage and API

A separate version-1 `organization.sqlite3` contains group metadata and memberships.
Creation/initialization and mutations use SQLite transactions. The original catalog
schema remains compatible; no new runtime dependencies are required. Deleting a
virtual collection cascades only its membership rows.

The local API uses the gallery's existing session, origin and Host restrictions:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET / POST | `/api/virtual-collections` | List for a source / create a group |
| PATCH / DELETE | `/api/virtual-collections/{id}` | Rename / delete with revision |
| POST / DELETE | `/api/virtual-collections/{id}/members` | Add / remove up to 500 IDs |
| GET | `/api/virtual-collections/{id}/images` | Browse available members with pagination |

List and browse take a `collection` query parameter. Mutations take `collection`
in JSON, defaulting to `personal`. Create/rename require `name`; other mutations
require `revision`, and membership actions require `image_ids`.

Existing text/similarity task endpoints accept optional `virtual_collection` and
`virtual_revision` fields. Upload tasks accept these as query parameters. Task
metadata retains scope identity/revision for reconnecting and restoring results.
Long-running searches keep their existing SSE progress and cancellation behavior.

See [validation evidence](benchmarks/2026-10-21-virtual-collections.md) for executed
checks and remaining platform limitations.
