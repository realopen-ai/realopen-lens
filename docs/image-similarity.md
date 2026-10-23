# Find visually similar images

Image queries search the visual embeddings of a saved collection using SigLIP2.
This is useful for finding related photographs, screenshots with similar layouts,
and visually similar documents or diagrams. It works offline after indexing.

For [library-wide similarity](library-search.md#image-similarity), add
`--all-folders`, or select **All folders** in the gallery before using **Similar**
or uploading an image. Mixed-precision library similarity remains experimental.
The examples below describe single-folder search.

## Command line

Use an image anywhere on your local disk:

```sh
uv run --no-sync lens similar /path/to/query.jpg \
  --index-dir runs/personal-index --limit 10
```

Or use the full `image_id` returned by `lens search` or `lens similar`:

```sh
uv run --no-sync lens similar --image-id FULL_IMAGE_ID \
  --index-dir runs/personal-index
```

Select a named collection with `--collection`. Results use the same JSON schema
as text search: rank, image ID, current source paths, verified thumbnail,
dimensions, OCR preview, visual similarity, processing status and issues.

By default the query image is excluded by its SHA256 content ID, including all
byte-identical copies. Add `--include-self` to return it. Re-encoded or resized
copies have different IDs. Published duplicate scans also exclude identical
decoded-pixel copies; approximate copies can still appear in results. Scores are cosine
similarities, not probabilities of relevance or proof of duplication.

The collection determines INT8 or FP32 precision automatically. An optional
`--siglip-precision int8` or `--siglip-precision fp32` asserts compatibility; it
never mixes vectors from different precisions. An FP32 example:

```sh
uv run --no-sync lens similar /path/to/query.png \
  --index-dir runs/personal-index-fp32 --siglip-precision fp32
```

## Python

```python
from pathlib import Path
from realopen_lens.search import search_similar

trace = {}
results = search_similar(
    Path("/path/to/query.jpg"),
    index_dir=Path("runs/personal-index"),
    limit=10,
    trace=trace,
)

# Supply exactly one of image or image_id.
results = search_similar(
    image_id="FULL_IMAGE_ID",
    index_dir=Path("runs/personal-index"),
    include_self=False,
)
```

The optional trace records the immutable generation, manifest checksum,
precision, query content ID, whether the embedding was reused, number of
searchable images and self-inclusion setting. Ties sort by image ID for
reproducible output.

## Reuse, resources and compatibility

An indexed image ID, or a path containing exactly the same bytes as an indexed
image, reuses the stored visual vector. This requires no model files or new
inference. An indexed ID can be queried even when its original source is missing;
returned results still require at least one unchanged source file.

A previously unseen image runs through the existing vision preprocessing: bounded
decode, EXIF orientation, transparency compositing and 224-by-224 SigLIP2 input.
Supported contents are JPEG/MPO, PNG, WebP, TIFF and BMP. The existing limits are
64 MiB encoded size and 24 million decoded pixels; animated/multipage inputs use
the first frame. A changed input file is rejected rather than returning a result
for unstable bytes. Metadata-only ctime changes are confirmed by rehashing when
size, modification time and file identity remain unchanged.

External inference checks the pinned vision graph checksum, registry,
preprocessing source and NumPy/Pillow/ONNX Runtime versions against the index.
It needs only the collection's vision graph, not E5, a text tokenizer, Tesseract
or a text model. Incompatible preprocessing/dependencies require reindexing.
Inference uses one disposable offline worker, 1–8 threads (default 2), and a
120-second deadline. Separate invocations reload the graph; there is no resident
image-query session yet.

Search reads the existing immutable generation and applies the same source and
thumbnail validation as text search. It neither changes the index nor copies or
moves original images. No index rebuild or additional dependencies are needed
for this feature when the existing encoder environment is compatible.

## Scope

This is visual similarity search. It does not apply OCR text fusion or the
experimental logistic reranker. Similar-looking forms with different numbers
may be close, while the same subject in a different layout may be distant.
Automatic near-duplicate grouping, similarity thresholds and relevance feedback
remain separate future work.

## Validation

The feature was smoke-tested against two saved 2,227-image collections, one INT8
and one FP32. Four inputs covered a photograph, a screenshot, a document and a
diagram. Each was queried by stored ID, exact local file and a separately encoded
90%-resolution variant: 24 requests in total. All 16 stored/exact-file requests
reused their vectors and excluded the query ID. All eight resized-image requests
returned the original at rank 1. Result thumbnails were inspected for both
precisions. These checks establish basic integration and transformation recovery;
they do not establish general image-query relevance or near-duplicate accuracy.

Individual results, source IDs, generation checksums, script and contact sheet
are retained locally under `runs/image-similarity-v1/` (ignored because they can
contain private collection paths). The installed CLI was also checked with both
precisions. Regression tests cover stable ranking, source validation, precision
compatibility, model integrity, worker deadlines/errors, invalid inputs and JSON
output. Native inference was exercised on macOS; Windows/Linux type checks do
not replace native inference validation on those platforms.
