# Public photograph sample: 2026-10-18

An initial real-image test used 50 randomly selected COCO photographs. This is a
small diagnostic sample, not a release benchmark or evidence of 8 GB support.

## Method

- Source: COCO 2014 Karpathy test split, sampled without replacement with seed 42.
- Downloads: 50 JPEGs, about 8.9 MB in total, with source URLs and SHA-256 hashes.
- Queries: one published English caption per image, plus assistant translations
  of the first ten captions into French and Arabic (70 queries in total).
- Relevance: the caption's source image is a known positive. Other images remain
  unjudged. Translations were not independently reviewed.
- Runtime: CPU-only INT8 SigLIP2 and multilingual E5, batch size one, two threads,
  three repeated query executions. Tesseract used `eng+fra+ara`.
- Host: Apple Silicon macOS machine with 16 GiB RAM.
- Network: inference completed under macOS `sandbox-exec` with network access
  denied. Model downloads and image downloads happened beforehand.

French and Arabic queries cover a smaller, overlapping subset of the English
queries; the language scores are not a controlled comparison between languages.
No training or feedback adaptation was performed.

## Retrieval results

Counts refer to the known source image appearing at rank one or within the first
five results, out of the number of queries in that row.

| Language | Queries | Visual rank 1 | Visual top 5 | Hybrid rank 1 | Hybrid top 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 50 | 39 | 48 | 1 | 47 |
| French | 10 | 9 | 9 | 1 | 9 |
| Arabic | 10 | 7 | 10 | 1 | 9 |

Visual retrieval found the known source image near the top for most captions.
The equal-weight reciprocal rank fusion baseline substantially reduced rank-one
quality. Sparse, noisy OCR-derived candidates received additional rank votes
regardless of whether OCR was useful for the query. This test therefore exposes
a fusion limitation that the generated text-image fixtures did not reveal.

The next ranking experiment should compare fixed fusion with query- and
signal-aware weighting or gating. Preserve the visual-only baseline and evaluate
photo and screenshot subsets separately before selecting an approach. Do not
tune solely to this sample and present the result as a held-out improvement.

## Resource measurements

The 70-query run took approximately 22.0 seconds for the measured pipeline and
sampled a peak of 729.4 MiB process-tree RSS. Vision inference processed about
13.6 images per second, excluding model loading. Memory sampling recorded no
access errors. Timings include the benchmark's staged execution and repeated
query encodings; they are not interactive end-to-end search latency.

The separate 50-English-query run took approximately 19.2 seconds and sampled
836.4 MiB peak RSS. Memory and timings vary between runs. RSS is sampled every
20 ms, may miss short spikes, and can count shared pages more than once.

## Reproduce the English sample

From a prepared installation with local models:

```sh
uv run --no-sync python -m benchmarks.samples --count 50 --seed 42
uv run --no-sync python -m benchmarks run --data data/public-sample --queries data/public-sample/queries.caption-derived.json --languages eng+fra+ara --corpus-kind public-coco-caption-derived
```

The downloader creates English queries only. The additional ad hoc translated
query file is saved locally as
`data/public-sample/queries.multilingual-provisional.json`; it is not distributed
with the repository. Source images, manifests, raw OCR, and reports remain in
ignored local folders. The original runs are in `runs/public-coco-50/` and
`runs/public-coco-multilingual/`.

See the [benchmark guide](../benchmark.md) for setup, offline verification, and
measurement definitions. See the [caption source](https://cs.stanford.edu/people/karpathy/deepimagesent/)
and [COCO](https://cocodataset.org/) for dataset context and source terms.

This photo-only test does not measure screenshot retrieval, OCR transcription
accuracy, model export parity, quantization loss, or performance on actual 8 GB
hardware. A larger, independently judged mixed collection is still required.
