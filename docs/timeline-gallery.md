# Timeline browsing

Open **Image library** to browse all indexed folders, select a folder for its own
images, or open a virtual collection for its saved members. Browsing uses indexed
metadata and does not load inference models. Library browsing keeps identical
decoded images together and retains their source locations.

**Newest first** is the default. **Oldest first** reverses the date order. Month and
day headings follow the complete ordered list, including across page boundaries.
Text and image searches continue to use relevance ranking; timeline controls and
headings are shown only while browsing.

## Where dates come from

- **Captured** uses the capture timestamp retained in image metadata. Its calendar
  day and clock time are kept as recorded, including when a timezone is unknown.
- **File modified** is a fallback when no usable capture date is recorded. It uses
  the indexed source signature, displayed and sorted in UTC. Copying or editing a
  file can change this timestamp; it is not evidence of when a photo was taken.
- **Date unknown** follows dated images in either sort direction.

Ordering follows these calendar timestamps, rather than converting capture times
with missing offsets into guessed global instants. Equal timestamps use stable
image IDs. For grouped copies, capture metadata takes priority; otherwise the
newest available copy's recorded modification date is used.

Older indexes can browse using file dates immediately. Refresh them to extract
capture metadata; compatible cached OCR and embeddings are reused.

## Loading and updates

The gallery requests 60 images at a time. Scrolling near the end loads the next
page; the button also supports manual loading or retrying a failed request.
Loading another page preserves selection and the loaded image navigation order.
Automatic paging pauses while image details or an import/collection dialog is open.
Changing scope or sort starts a fresh list and clears selection.

Pages carry generation and timeline snapshot identifiers. A changed index,
unavailable image that affects the list, or edited virtual collection produces a
conflict rather than silently skipping or repeating images. **Reload images**
starts a fresh timeline. Automatic folder refresh retains the current library
scope and selected date order; existing protections against replacing an active
selection or open viewer remain in effect.

## API

Use `GET /api/collections/{collection}/images`, `GET /api/library/images`, or
`GET /api/virtual-collections/{id}/images` with `order=newest|oldest`, `offset`, and
`limit` (1–100). Virtual collections retain their existing scope/revision parameters.

Responses add `timeline` to each browse result, plus `order`, `snapshot` and
`next_offset`. Pass the returned `generation` and `snapshot` with the next request,
and preserve order and virtual revision. `next_offset: null` ends the list.
Conflicts return HTTP 409. Search responses and relevance ordering are unchanged.

There is no date-jump rail or DOM virtualization in this version. Paging bounds
each response; images already loaded stay in the page for selection and navigation.
