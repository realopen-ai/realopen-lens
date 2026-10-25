# Metadata search validation

This branch adds EXIF/GPS extraction, HEIC decoding and deterministic natural-language
constraints to production search. It does not change embedding weights, OCR models
or the ranking formula for unconstrained descriptions.

## Automated checks

- 821 Python tests passed; one optional test skipped.
- 92 desktop/mobile browser tests passed, including interpretation/literal reruns,
  camera/capture/GPS display, navigation and existing workspace workflows.
- Black, Flake8, Ruff and mypy native/Windows/Linux checks passed.
- Real JPEG and HEIC fixtures retain camera model, capture offset, exposure and GPS.
- A synthetic 56-image candidate test puts the only camera match among 55 other
  images and verifies filtering before candidate limits and encoding only `sunset`.
- Filtered feedback, hidden browsing, virtual scopes, legacy backfill and training
  exclusion have regression coverage.

The committed metadata fixtures contain synthetic values and no personal photo
information. Their generation/provenance is described in `tests/fixtures/metadata/`.

## Frozen macOS execution

The previous app indexed five generated JPEG controls with real Tesseract and
SigLIP2/E5. Controls vary camera model, capture year, Paris/Tokyo GPS coordinates,
and absent EXIF. The new frozen app refreshed that index and then indexed a new
HEIC control. No user workspace or running indexing job was changed.

| Operation | Processed metadata / OCR / visual / semantic | Reused metadata / OCR / visual / semantic |
| --- | --- | --- |
| Previous app, first five-image index | 5 / 5 / 5 / 5 | 0 / 0 / 0 / 0 |
| New app, metadata backfill | 5 / 0 / 0 / 0 | 0 / 5 / 5 / 5 |
| Add one HEIC | 1 / 1 / 1 / 1 | 5 / 5 / 5 / 5 |

The plain `orange` query returned the same image-ID ordering before and after
backfill, on the unchanged five-image corpus. This is a functional regression
check, not a broad retrieval-quality evaluation.

| Native request | Expected eligible results | Actual |
| --- | ---: | ---: |
| Camera: iPhone 11 Pro Max | 4 | 4 |
| Camera + June 2024 + Paris radius | 2 | 2 |
| Visual `orange` + camera/date/Paris constraints | 2 | 2 |
| GPS near Tokyo coordinates | 1 | 1 |
| HEIC format | 1 | 1 |
| Missing camera model | 0 | 0 |

A second FP32 folder contained a copy already indexed in INT8. Library search
with the camera/date/Paris constraints returned two eligible groups. A separate
`orange in Paris, France` run returned four eligible groups across both precisions
and retained the existing experimental mixed-precision diagnostic. Exact-content
copies did not receive extra votes.

Whole-CLI timings and individual image IDs are retained in the local native JSON
records. They include startup, model verification/loading and lazy city-index loading;
they do not establish isolated latency or low-memory hardware guarantees. These
controls have valid metadata ground truth, not independently reviewed visual labels.
No semantic accuracy score is inferred from their successful execution.

## Interpretation failure analysis

An initial scan found false city matches from short alternate names (`API`, `UNE`)
and place names printed in hotel advertisements. Short noncanonical abbreviations
are now excluded from place lookup, and document-place references remain content
unless capture-location wording is explicit. Regression cases cover both causes.

After these corrections, all 357 frozen retrieval-v2 queries retained their original
search text with no inferred metadata constraints or interpretation exceptions.
This was a parser regression scan; the 357-query retrieval benchmark was not rerun.

A simulated HEIC decoder-version upgrade also exposed premature cache restoration
before metadata identified the container. Non-metadata reuse now waits for required
metadata, and full-pipeline fallback checks the producer's decoder version for HEIC.
The regression verifies one HEIC analysis is recomputed while compatible PNG OCR/
visual/semantic analyses stay cached.

City queries use a bounded radius and an offline GeoNames dataset of 34,188 cities.
The compressed dataset is 3,078,832 bytes. One development-process parser scan
observed about 94 MiB RSS growth while lazily loading names/aliases; this is an
observed host measurement, not a maximum memory bound.

## Packaging and retained evidence

The macOS archive includes the decode-only `pi-heif` runtime, its native dependency
notices, and the offline city data. SHA256 verification and all 12 frozen-package
checks passed. An isolated real frozen-gallery/browser run searched the full
camera/date/Paris phrase, received two results, displayed camera/GPS fields and
rendered a HEIC thumbnail fallback successfully.

Local artifacts are preserved under `runs/metadata-search-v1/`: per-stage native
baseline/final records, mixed-query records, commands/logs, fixture corpus, parser
scan, screenshots, full/targeted Python logs, browser log and package checks.
The native scripts in that directory document the executed local experiment;
portable regression coverage is committed in `tests/test_metadata_search.py`,
`tests/test_local_api.py` and the gallery/browser tests.

Remaining limits: supported phrase patterns rather than unrestricted language
understanding; city-radius matching rather than country/city polygons; incomplete
small-place coverage; no video/Live Photo motion indexing; unknown EXIF stays
unknown. Filter-aware learned-ranking evaluation remains separate. Windows/Linux
are covered by decoder wheels, tests and type checks; no new native builds on
those platforms were executed for this branch.


## Screenshot-marker follow-up

EXIF ASCII/Unicode UserComment, ImageDescription, PNG text and bounded XMP Dublin
Core descriptions now produce `marked` / `unknown` evidence with provenance.
Broad screenshot requests can include marker candidates outside the normal channel
pool; unmarked files remain eligible. Scene-specific screenshot requests preserve
content ranking. Explicit `marked screenshots` requests filter by marker evidence.
Existing exact-query feedback takes precedence and keeps its namespace for broad
marker preference; marker-only requests have a separate constraint context.

Sixteen screenshot regressions include malformed/oversized/entity-declaring XMP,
wrong XML namespaces, Unicode comment encodings, filenames/dimensions without
markers, candidate inclusion, library scopes, feedback preservation and both real
JPEG/HEIC decoder fixtures. All 821 Python tests and 92 browser tests passed.

A macOS `sips` conversion of a UserComment-only JPEG stripped that marker but kept
the capture timestamp. The resulting HEIC stays unknown. A second conversion with
ImageDescription retained the marker in XMP.dc.description, and provenance reports
that source rather than inventing an EXIF UserComment. Both outcomes are committed
as synthetic fixtures. These are conversion tests, not an iPhone AirDrop validation.

The frozen app backfilled six existing images with six metadata attempts and zero
OCR/visual/semantic attempts. Adding two marked files and one unknown control
processed only those three new images. Native marker-only folder/library requests
returned two files; the broad previous-week query retained all three eligible
files, with explicit markers first. Detailed records remain in
`runs/metadata-search-v1/screenshot-native.json` and related screenshot logs.
