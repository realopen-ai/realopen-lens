# Bounded search pagination — 2026-10-26

Native diagnostic on an existing FP32 index with 2,227 indexed image entries.
The library API run covered the same single folder; it is not a native multi-folder
or mixed-precision scaling claim. Regression tests cover mixed-precision scopes.

Eight cases traversed 20 pages each, retrieving 1,000 distinct groups per case.
All ordinary top-50 result records matched the paginated first page exactly.
No image IDs or duplicate aliases repeated across pages. Counts describe retrieval
coverage, not correct/relevant results. No qrels were supplied and no new Recall@50,
MRR or nDCG score is claimed.

| Case | Ordinary first 50 (s) | Prepare paginated pool (s) | Later-page P95 (ms) | Groups |
| --- | ---: | ---: | ---: | ---: |
| folder-search-0 | 5.526 | 3.281 | 27.2 | 1000 |
| folder-search-1 | 0.890 | 1.758 | 26.8 | 1000 |
| folder-search-2 | 0.746 | 1.780 | 28.6 | 1000 |
| folder-similar-3 | 0.487 | 1.008 | 29.8 | 1000 |
| library-search-0 | 5.410 | 2.892 | 40.3 | 1000 |
| library-search-1 | 0.866 | 3.168 | 35.6 | 1000 |
| library-search-2 | 0.847 | 3.014 | 62.0 | 1000 |
| library-similar-3 | 2.152 | 6.023 | 37.6 | 1000 |

Maximum sampled coordinator/child RSS was 1680 MiB,
with 0 sampling access errors. Samples were taken every 20 ms;
shared pages can count twice and short peaks can be missed. Initial ordinary calls
include cold encoder startup while subsequent pool preparation can reuse encoders.
The sequential observations are not randomized, repeated or CPU-isolated; they
must not be read as evidence that a deeper initial search is faster. Pool preparation
performs two retrievals; continuation performs validation and slicing without inference.

Raw configuration, per-case results, every continuation latency and memory samples
remain under `runs/search-pagination-v1/native/`. The reproducible runner is
`benchmarks/experiments/search_pagination.py`. Public JSON retains aggregate cases
without source image paths. See the [behavior and limits](../search-pagination.md).

## Validation

- Full Python suite: 886 passed, one optional skip. After adding further feedback
  export and removal regressions, the affected paging/library suite passed all
  34 cases.
- Full gallery suite: 120 desktop/mobile checks passed, including later-page
  feedback, similarity paging, stale cursors and continuing after hiding every
  visible result. Screenshots inspected without horizontal overflow.
- Python formatting/lint, mypy native/Windows/Linux targets, TypeScript, Prettier
  and Vite checks passed. The macOS archive passed 12 frozen-package smoke checks.
- Native model-bearing runs are macOS FP32 observations. Windows/Linux native
  packaging and model-bearing mixed-precision multi-folder paging were not run.
