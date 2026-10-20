# Full saved-collection SigLIP2 precision evaluation

Date: 2026-10-20. Production INT8/FP32 comparison on the same saved collection.

## Confirmed collection coverage

The audited FP32 collection is a full production index. It contains all
**2,052 frozen reference images**, plus **175 additional unreviewed images**:
2,227 searchable unique images across 5,729 paths. The original scan considered
5,744 paths, with 15 expected input-safety rejections and one degraded partial-OCR
newspaper. None of the reference images is missing. The original FP32 generation
was preserved; no FP32 reindexing was needed.

Both policies search all 2,227 images. Additional images remain unreviewed and are
not training negatives. The 119 query families contribute 357 previously inspected
queries, 119 each in English, French and Arabic; this is not a fresh holdout.

## Controlled method

An isolated INT8 control reuses verified metadata, Tesseract OCR and INT8 E5
analyses, then recomputes all 2,227 visual embeddings through production indexing.
Published members, statuses, OCR records and E5 vectors match exactly. Model/runtime
specifications differ only in SigLIP2 precision. All 2,052 reference FP32 image
vectors also match the earlier cached FP32 diagnostic bit-for-bit. Across all 357
paired native requests, semantic and keyword channel rankings are unchanged.

Every request executes production hybrid search, including integrity checks,
encoder subprocess startup and inference. Both policies eventually obtain results
for all 357 queries, with initial failures and retries retained below. Default 50-per-channel
candidate budgets, two CPU threads and INT8 E5 are fixed. There are zero memory
sampling access errors.

## Execution reliability and retries

| Policy | First-pass successes | First-pass failures | Total attempts | Final successful queries |
|---|---:|---:|---:|---:|
| int8 | 357 | 0 | 357 | 357 |
| fp32 | 356 | 1 | 358 | 357 |

The FP32 `v2-inkscape-cube-en` request exceeded the 120-second encoder deadline
after 123.207 seconds. It succeeded on its only retry with unchanged settings.
Initial failed requests are preserved in the attempts table and structured evidence.
Only failed queries are retried with identical production settings and deadlines.
Ranking-quality tables below use final successful results. First-pass deadline-aware
quality (failure means no returned results) is separately retained in JSON.
Latency tables use every first-pass request, including errors/timeouts; all-attempt
and latest-success timing distributions are retained separately.

## Retrieval quality

| Policy | Pool recall | Recall@50 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| int8 | 0.9398 | 0.9194 | 0.5540 | 0.6006 |
| fp32 | 0.9908 | 0.9899 | 0.8293 | 0.8580 |

Metrics are query-macro averages against known frozen judgments. Full-pool
recall is separate from final hybrid Recall@50. Unjudged results can be relevant;
known-judgment rank degradation does not prove every competing result is wrong.

| Language | INT8 R@50 | FP32 R@50 | INT8 MRR@10 | FP32 MRR@10 |
|---|---:|---:|---:|---:|
| en | 0.9605 | 0.9983 | 0.6655 | 0.8696 |
| fr | 0.9622 | 0.9966 | 0.5900 | 0.8761 |
| ar | 0.8354 | 0.9748 | 0.4064 | 0.7421 |

| Query kind | INT8 R@50 | FP32 R@50 | INT8 nDCG@10 | FP32 nDCG@10 |
|---|---:|---:|---:|---:|
| photos | 0.8595 | 1.0000 | 0.4538 | 0.8837 |
| screenshots | 0.9048 | 0.9810 | 0.6528 | 0.8056 |
| documents | 0.9599 | 0.9856 | 0.6559 | 0.8881 |
| diagrams | 1.0000 | 1.0000 | 0.6872 | 0.8466 |

The larger inventory and updated OCR differ from older 2,052-image runs. These
two matched policies are the controlled comparison; do not mix historical scores.

## Latency and resources

| Policy | Mean s | Median s | P95 s | P99 s | Peak sampled tree MiB |
|---|---:|---:|---:|---:|---:|
| int8 | 2.019 | 2.036 | 2.313 | 3.293 | 1186.0 |
| fp32 | 7.764 | 3.682 | 18.104 | 22.645 | 3009.0 |

The machine is an 8-core arm64 Mac with 16 GiB RAM. During FP32, a two-second
sample showed battery operation, roughly 10.6 GiB swap in use, 18.25 MiB/s swap-in
and 1.66 MiB/s swap-out. Only one encoder child was active in the sampled tree;
there was no observed leaked-worker buildup. Cold latency fluctuated substantially.
The recorded timings describe this workload; they are not a controlled claim
that precision alone caused the entire timing difference. Failed attempts remain
in the first-pass latency distributions and are not replaced by retry timings.

Repeated model verification/startup is a concrete optimization target. The current
CLI reloads encoders per request; profiling a bounded resident worker and controlled
AC-power/low-pressure cold/warm runs should precede performance default decisions.
Native Windows/Linux inference and an 8 GiB machine were not tested.

## Inspected failures and remaining weaknesses

- The fisheye dining-room target (`photos-03-en/ar`) moves from absent INT8
  hybrid top 50 to FP32 ranks 1/2. Two stopped motorcyclists (`photos-05-en/ar`)
  move from absent to ranks 2/1. Original images match their frozen captions.
- Arabic Form 4868 is keyword rank 1 in both policies, but its sole matching
  token is the bare number 4868. The existing evidence rule requires two
  matched terms or an exact code/quoted phrase, so lexical evidence is zero.
  FP32 improves its hybrid rank from absent to 9 but does not repair that gate.
- `operations-reports-ar` has 20 known positives. INT8 retrieves 5 in top 50;
  FP32 retrieves none and none enter its candidate pool. All 20 have readable
  English OCR (554–591 characters, one passage), so missing OCR or truncation
  does not explain this. English/French translations retrieve targets. This
  known-target cross-language candidate gap cannot be fixed by reranking alone.
- Arabic FileZilla (`v2-filezilla-ftp-ar`) moves from hybrid rank 9 to 18
  (visual rank 8 to 17), dropping out of top 10. The original French FTP
  screenshot matches the reference; neither OCR channel retrieves it.
- There are 6 known-judgment nDCG regressions and 2 Recall@50 regressions;
  their query IDs, metrics and positive ranks are retained in structured evidence.
  These are diagnostic regressions under incomplete relevance judgments.

Input rejections are expected stress fixtures: 9 paths above the 24-million-pixel
cap, one decompression bomb, two invalid images, two truncated files and one
unsupported GIF. These are separate from query execution failures and retries.
The newspaper retains explicit `PartialOCR` degraded status in both indexes.

## Artifacts and reproduction

The [structured evidence](2026-10-20-full-siglip2-precision.json) retains all 357
paired known-query metrics and positive ranks, regression lists, configurations
and diagnostic context without publishing personal extra paths or images.
Use the [saved collection guide](../saved-precision-evaluation.md) for commands.
Local artifacts remain under `runs/fp32-full-collection-v1/`: both evaluations
with individual SQLite/JSONL records, candidate scores/labels, resource samples,
source snapshots, control preparation, verification records and logs. Identical
commands resume; changed sources/configurations require a new run directory.

## Validation and next steps

Production defaults and ranking rules are unchanged. Keep FP32 opt-in. Address
cold-start/resource costs and the observed numeric/cross-language candidate gaps
with separate regression studies. Precision-aware reranker training must use
reviewed exact-corpus data or explicitly established provenance/split boundaries
for additional images; current reranker artifacts remain INT8-only.

Validation: 280 tests passed; Black, Flake8 and Ruff passed; mypy passed for
the native, Windows and Linux targets (120 source files). These checks do not
replace native inference testing on Windows/Linux.
