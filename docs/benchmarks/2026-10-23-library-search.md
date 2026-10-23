# Library-wide search: mixed-precision diagnostic

The first global search implementation completed all 357 EN/FR/AR queries without
errors. FP32-only search has the highest known-judgment ranking quality. Mixed
search improves on INT8-only overall, but raw cross-precision score merging also
introduces regressions and model switching increases latency. Mixed visual/hybrid
ranking remains explicitly experimental; homogeneous libraries avoid cross-precision
score comparisons.

## Method

The controls run production single-collection hybrid search against existing
INT8 and FP32 snapshots of the same 2,227 byte-unique images. The mixed library
uses disjoint SHA-256-modulo partitions from those snapshots, with one partition
per precision. It searches 2,225 available exact-content groups after grouping.
No source images or original indexes are modified or re-encoded.

All runs use the same 357 queries, judgment-file checksum, ranking source hashes,
dependency versions and two inference threads. Query memory and learned rerankers
are disabled. Each run retains 357 successful individual records with zero
process-memory sampling access errors.

These are existing development judgments, including agent-authored translations,
not an independently reviewed holdout. Recall@50 below is computed on returned
results, rather than the full candidate union before fusion. Candidate selection
and duplicate presentation differ between single-collection and global search;
this is a system comparison, not an isolated quantization experiment. Host load
was not isolated, so latency comparisons are diagnostic.

## Results

| Metric | INT8-only | FP32-only | Mixed library |
| --- | ---: | ---: | ---: |
| Queries / failures | 357 / 0 | 357 / 0 | 357 / 0 |
| Known-judgment Recall@50 | 0.9222 | 0.9899 | 0.9562 |
| MRR@10 | 0.5619 | 0.8331 | 0.6776 |
| nDCG@10 | 0.6081 | 0.8610 | 0.7194 |
| Median seconds | 0.3342 | 0.3910 | 1.7079 |
| P95 seconds | 1.1206 | 0.5442 | 2.8076 |
| Peak sampled process-tree RSS, MiB | 1408.0 | 2277.8 | 3055.4 |

RSS includes the coordinator and owned subprocesses. Shared pages may count more
than once and sampling can miss brief spikes. Median mixed latency is about 4.37
times the FP32 control. Only one query worker is resident at a time, but switching
precision reloads model sessions; the first precision can also load the E5 model.

Against INT8, mixed nDCG@10 improves for 102 queries, ties for 212, and regresses
for 43. Against FP32, it improves for 11, ties for 229, and regresses for 117.

## Diagnosed examples

- `photos-01-en`: the relevant armored-person/horse image ranks 29 with INT8,
  1 with FP32, and 20 in mixed search. Its mixed partition uses INT8. Its visual
  cosine is 0.03895 with INT8 versus 0.19450 with FP32. The global merger cannot
  recover information that quantization has already lost.
- `v2-al-hoda-ar`: the relevant newspaper ranks 1 in **both** homogeneous controls
  but 5 in mixed search. Its INT8 cosine is 0.13842. FP32-partition distractors
  score up to 0.17610 and move ahead when raw similarities are merged. OCR evidence
  is zero for these candidates. This is direct evidence of a cross-precision
  ordering problem beyond the target's homogeneous retrieval quality.

The second example means aggregate mixed quality falling between the controls is
not sufficient evidence that the merger is correct. Scores from the two exports
cannot simply be assumed to have equivalent relevance scales.

## Replay experiments and decision

A subsequent full-score replay used existing cached embeddings for the exact same
357 query texts. Their encoder source checksum matches the current implementation;
probe outputs and input inventory checksums are verified. All 357 raw baseline
rankings were reproduced exactly and candidate visual/semantic/hybrid scores matched
within 1e-6. No new model inference or control benchmark was required.

The frozen mixed corpus was replayed with four label-free alternatives. Moment
normalization maps INT8 mean/standard deviation to the FP32 distribution; robust
normalization uses median and median absolute deviation. Group ranking interleaves
native hybrid ranks within precision groups, using raw scores only for rank ties.
The evidence-first variant adds a hard OCR-evidence priority. Each precision group
contains all of its folders, not one vote per folder.

| Policy | Overall Recall@50 | Overall nDCG@10 | Validation Recall@50 | Validation nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Raw scores | 0.9562 | 0.7194 | 0.9667 | 0.6614 |
| Moment normalization | 0.9563 | 0.7217 | 0.9556 | 0.6606 |
| Robust normalization | 0.9504 | 0.7115 | 0.9556 | 0.6351 |
| Group ranking | 0.9620 | 0.7192 | 0.9556 | 0.6458 |
| Group ranking, evidence first | 0.9620 | 0.6827 | 0.9556 | 0.5989 |

None establishes a validation improvement over the raw baseline. Group ranking
raises overall recall but lowers validation ranking quality; normalization also
changes relative positions without a consistent benefit. **No alternative is
promoted.** This avoids treating a fix for one diagnosed example as a general gain.
The existing splits are diagnostic development splits, not a newly independent test.

The mixed merge remains available with an experimental marker in the compact
ranking caption and API provenance. Homogeneous search keeps its existing policy.
The shared RAM-based default now applies across adapters, making new collections
consistent by default; existing indexes, imports and explicit overrides can still
produce mixed libraries. This does not repair quantization losses or certify score
calibration. Replay timings exclude query inference and are not runtime benchmarks.

Real INT8/FP32 API smoke also passed: 50 results included both precisions, observable
progress/SSE, source thumbnail serving, explicit feedback and hidden-image review,
media after a folder rename, and task cancellation. Query-memory regression coverage
includes changing a judgment when only a different-format exact alias remains.

## Next work

Further evaluate precision-aware ranking against the retained raw-score baseline,
using a global pool within each precision rather than assigning votes per folder.
Keep substantive lexical/semantic OCR evidence and test unequal group sizes;
normalizing unrelated results to a group's top rank can create a different bias.
The evaluated alternatives above remain diagnostic; broader distributions and
independent relevance review are needed before selecting a calibrated replacement.

Measure model-switching costs separately from retrieval and source validation.
Retain bounded model residency; do not solve latency by silently loading an
unbounded set of models. Homogeneous libraries should reuse their resident worker.

The implementation, its regressions, and the remaining merge decision are tracked
separately from any claim of improved independent relevance quality.

## Evidence

Raw records and configurations remain in the ignored local directories:

- `runs/library-search-v1/user-control-int8-357/`
- `runs/library-search-v1/user-control-fp32-357/`
- `runs/library-search-v1/validated-mixed-357/`
- `runs/library-search-v1/merge-replay-01/` — complete per-query replay rankings,
  full raw score arrays, probe provenance and split summaries.
- `runs/library-search-v1/native-api.json` — real-model API integration evidence.
- `runs/library-search-v1/comparison.json` — checked run comparability and every
  per-query metric delta.

The configurable runner is `benchmarks/experiments/library_search.py`; user-facing
behavior and limitations are described in [Library search](../library-search.md).


## Image-similarity extension

Library image queries were checked separately from the 357 text queries above.
Two real indexed queries, one from each precision partition of the saved mixed
library, returned 50 results each and excluded the query's exact-content group.
Each query reused its stored vector and encoded the missing precision with the
actual vision model. Same-precision result scores matched the existing folder
similarity path within `1e-5`. A native API job also completed with 50 results and
SSE delivery. Individual outputs, source provenance, progress phases and score
comparisons are retained locally in `runs/library-search-v1/similarity-native.json`.

These are functional checks, not independent image-query relevance judgments or
a replacement for the text-query metrics. Regression coverage additionally checks
single-folder rank/score parity without models, cross-format self-exclusion,
external image encoding, file mutation, cancellation, recording, API uploads and
CLI routing. Mixed-precision similarity remains explicitly experimental.
