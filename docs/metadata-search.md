# Natural-language metadata search

Type a complete request into the normal search box. Lens extracts recognized
metadata constraints locally, narrows the eligible images, and searches the
remaining scene/text description using the existing retrieval channels. No extra
language model or online geocoding service is used. A search with only metadata
constraints does not load the image/text encoders.

| Request | Interpretation |
| --- | --- |
| `sunset taken with my iPhone 11 Pro Max` | Camera model matches; visually search for sunset |
| `photos taken with iPhone 11 Pro Max in June 2024 in Paris, France` | Camera, calendar dates and a radius around Paris |
| `photos within 5 km of Paris, France` | GPS coordinates within the stated city-centre radius |
| `photos near 48.8566, 2.3522` | GPS coordinates within 25 km of that point |
| `screenshots from last week` | Search for screenshots within the previous Monday–Sunday interval |
| `PNG diagrams at least 2000 pixels wide` | Format and width constraints; search for diagrams |
| `JPEG photos larger than 2 MB` | JPEG files with at least 2 MiB encoded size |
| `photos from folder "Holiday"` | Narrow the current search scope to that folder name |

The gallery shows the interpretation beside the results. **Use entire query as
search text** reruns a request without metadata constraints. The CLI equivalent
is `lens search QUERY --literal`. CLI results remain JSON on stdout; interpretation
is reported on stderr. The HTTP search input accepts `interpret_metadata: false`.

Recognition is deterministic and phrase-based, not unrestricted language
understanding. Camera/date phrases include English, French and Arabic patterns;
city aliases come from GeoNames. Unknown descriptions remain visual/OCR content.
Places mentioned in document/advertisement descriptions stay content unless the
request explicitly asks where the image was captured. Ambiguous city names need
country/region clarification unless indexed GPS points identify exactly one city.

## Retained metadata

The metadata stage extracts a bounded, JSON-safe subset of EXIF:

- Camera make/model, lens make/model, orientation.
- Original capture timestamp, digitization timestamp and EXIF modification time.
- Recorded capture offset, ISO, exposure, aperture and focal length.
- GPS latitude, longitude and altitude, including hemisphere/altitude signs.

Existing file metadata retains format, dimensions, frame count and encoded size.
File modification dates are evaluated per available source path; copies can have
different filesystem dates. Metadata is displayed in the image viewer. Original
files are read only and retain their complete embedded metadata.

Unknown, malformed or stripped EXIF values stay unknown. Partial GPS pairs and
non-finite numbers are not used. Camera/location constraints exclude images with
missing fields. Lens never infers a camera or GPS location from image appearance.

Explicit `taken in/on` wording or a camera-and-date request requires a capture timestamp. Generic date
wording uses capture date when present and otherwise source modification date;
the interpretation states this fallback. Capture dates use the recorded calendar
date, preserving any recorded offset; missing offsets do not become guessed UTC.
Relative dates use the server's local calendar. Missing metadata is not evidence
that a photo was taken outside a date range or city.

## Offline places

The bundled GeoNames cities15000 index contains 34,188 cities and alternate names,
plus country names for disambiguation. Its compressed JSON occupies about 3 MB.
It loads lazily only when a query needs place lookup. No location request leaves
the device. Data attribution/checksums/reproduction instructions are in
[`metadata/data/README.md`](../src/realopen_lens/metadata/data/README.md).

City matching uses a displayed radius (25 km by default), not municipal borders.
Country-only polygons, addresses, small settlements absent from the dataset,
landmarks and online maps are not provided. Coordinates with an explicit radius
work without a named city. GPS metadata may be absent after editing/exporting.

## HEIC and existing indexes

JPEG, PNG, WebP, TIFF and BMP support remains; HEIC/HEIF primary still images are
now decoded using the `pi-heif` Pillow plugin. HEIC sequences/Live Photo video
components are not indexed as videos. Browsers that cannot display the HEIC
original show the indexed thumbnail, and **Open original** uses the system viewer.

Refresh a folder to backfill EXIF into its published generation. The reviewed
pre-metadata worker/encoder signatures are recognized: metadata and thumbnails
refresh while compatible OCR, visual vectors and semantic vectors are reused.
Adding a HEIC processes only that new image. Unknown legacy implementation hashes
remain conservatively incompatible. Real changes to processing, decoder versions,
models or OCR inputs still invalidate affected caches.

Metadata filtering happens before channel candidate limits and duplicate collapse,
inside the current folder/library/virtual-collection scope. For exact pixel copies
with different metadata, eligible source representations are selected before
library fusion. Metadata-filtered queries keep the selected visual/OCR/hybrid mode; learned
rerankers require separate filtered-query evaluation before use here.

Saved searches retain the original request and interpreted constraints. Exact-query
memory includes the constraint context, so literal and filtered requests, or
relative-date requests resolved to different periods, do not overwrite each other's
judgments. Existing labels remain stored. Filtered feedback is excluded from the
current unfiltered reranker-training review until a filter-aware evaluation exists.

Validation evidence: [metadata search study](benchmarks/2026-10-25-metadata-search.md).

## Screenshot markers

Lens reads explicit embedded `Screenshot` / `Screen Capture` values from EXIF
UserComment or ImageDescription, PNG Description/Comment text, and XMP Dublin Core
description (including RDF language alternatives and compact attributes).
EXIF ASCII/Unicode comment headers are decoded; XMP parsing is bounded and rejects
DTD/entity declarations. Ordinary prose mentioning screenshots does not count.

The viewer shows **Screenshot marker: Present / Not recorded**, with evidence
sources when a marker is present. This records a marker, not verified capture
history: metadata can be edited. Filename, PNG format, screen dimensions and lack
of camera EXIF do not become screenshot labels. Missing markers remain unknown.

- `screenshots` or `screenshots from last week`: broadly requested screenshot
  results prioritize explicit markers within the current metadata/date scope.
  Unmarked images stay eligible for existing visual/OCR retrieval.
- `screenshots of invoices`: the scene/content query keeps its normal ranking;
  an unrelated image is not promoted solely because it has a marker.
- `marked screenshots` or `marked screenshots from last week`: explicitly request
  only marked files. With no remaining scene text this needs no encoder inference.

Broad marker preference preserves the existing exact-query feedback namespace;
explicit relevant/irrelevant decisions take precedence. Marker-only requests have
their own metadata constraint context. Marker candidates are included independently
of channel top-50 pools, after scope/date eligibility and before final result limits.

Refresh folders to backfill the new fields; compatible OCR and embeddings remain
cached. Photos-library `photoScreenshot` state and macOS filesystem/Spotlight flags
are not imported. Synthetic JPEG/HEIC tests cover both preserved XMP descriptions and a conversion
that dropped UserComment while retaining the date. A real iPhone AirDrop transfer
was not tested. If a transfer strips the marker,
Lens keeps the file unknown and uses the available content-based retrieval.
