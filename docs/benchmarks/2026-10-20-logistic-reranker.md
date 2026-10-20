# Logistic reranker: first reviewed experiment

Date: 2026-10-20. Status: **experimental; hybrid remains the default**.

The first logistic model improves average known-judgment MRR@10 and nDCG@10 on
the fitting-held-out test split, but suppresses relevant reports and some French
screenshots. Test top-50 recall regresses. The nDCG improvement is uncertain with
only 11 positive source/duplicate components and incomplete agent-reviewed labels.
These results support an opt-in experiment, not automatic promotion or a claim of
general superiority. No feature or parameter was tuned in response to test results.

See the [reproduction guide](../logistic-reranker.md) and
[structured report](2026-10-20-logistic-reranker.json).

## Data and review

The same published Tesseract collection contains 2,000 unique images and paths.
Its frozen 180 queries cover 60 intent families in English, French, and Arabic.
Train/validation/test contain 108/30/42 queries; source documents, duplicate groups,
and translated query families remain indivisible across splits.

Before fitting, 125 additional candidate images were inspected in five contact
sheets, with full-resolution checks of added positives and a related bank diagram.
For each training translation, review selected up to three same-split unjudged
candidates within the original hybrid top 10, then combined pairs by family/image.
All 212 pair decisions and observed evidence are frozen in
[`training-review-v1.json`](../../benchmarks/sources/retrieval-v1/training-review-v1.json).

| Decision | Family/image pairs |
|---|---:|
| Directly relevant | 4 |
| Explicitly irrelevant to required content | 165 |
| Ambiguous, still unjudged | 43 |

Added positives are the high-resolution Form 1065, the alpha-channel spectrogram,
and two other white toilets satisfying the raised-lid/hole description. Decisions
are shared across translations: 507 judgments are added, with 429 additional labeled
pairs present in the saved candidate pools. Training grows from 923 pairs
(104 positive) to **1,352 pairs (116 positive, 1,236 negative)**.

The canonical judgment file is unchanged. Validation and test query objects and
judgments are exactly unchanged, with no label changes. Added
training positives increase positive training source components from 16 to 18;
validation/test retain seven/11. Unknowns never become implicit negative labels.
Review was by Codex, without independent human or native-speaker verification.

## Model and selection

The model has 14 explicitly allowed score/rank, missing-evidence, OCR-length, and
interaction features. No identity, category, split, language, or label enters the
feature vector. Training-only means/scales standardize features. Each family and
translation receives equal mass, with classes balanced within a query.

Training uses NumPy and bounded deterministic Newton optimization of weighted
binary logistic loss with L2 regularization and an unpenalized intercept. It adds
no runtime or training dependency. The selected fit converged in six updates with
maximum absolute gradient `2.80e-8`. Scores are uncalibrated logits, used only to
reorder the existing union of up to 150 candidates.

| L2 | Validation MRR@10 | Validation nDCG@10 |
|---|---:|---:|
| 0.001 | 0.5459 | 0.5894 |
| **0.01** | **0.5514** | **0.5938** |
| 0.1 | 0.5151 | 0.5667 |
| 1 | 0.5365 | 0.5896 |

The fixed grid and tie rules were established before test evaluation. Selection
uses validation nDCG, then MRR, then stronger regularization. The selected JSON
and checksum are saved before evaluating the test split. Its original coefficients,
scaling, intercept, and regularization remain unchanged after failure inspection.
Repeating final training produced a byte-identical model.

## Paired ranking results

Both rankings use identical candidates and the same reviewed references. Metrics
are query-macro averages over known judgments; they are not classification accuracy
or exhaustive retrieval quality. Training results use the added training labels,
so their aggregates should not be compared directly with the earlier study's
original-label training or overall aggregates. Held-out references are unchanged.

| Split | Queries | Hybrid MRR@10 | Logistic MRR@10 | Hybrid nDCG@10 | Logistic nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Train | 108 | 0.6512 | 0.7083 | 0.7024 | 0.7319 |
| Validation | 30 | 0.4661 | 0.5514 | 0.5151 | 0.5938 |
| Test | 42 | 0.6675 | 0.7738 | 0.6945 | 0.7502 |

| Test language | Queries | Hybrid MRR@10 | Logistic MRR@10 | Hybrid nDCG@10 | Logistic nDCG@10 |
|---|---:|---:|---:|---:|---:|
| English | 14 | 0.8810 | 0.9286 | 0.8798 | 0.8987 |
| French | 14 | 0.6190 | 0.7143 | 0.6590 | 0.6933 |
| Arabic | 14 | 0.5024 | 0.6786 | 0.5448 | 0.6585 |

Test top-50 known-positive recall drops from **0.8762 to 0.8036**. The full-union
test recall is **0.8833 for both methods**. Reordering a union larger than 50 can
change which positives survive the cutoff; only union recall is invariant.
Test judged fraction at 10 falls from 0.2167 to 0.1452, so more returned items have
unknown relevance, not confirmed negative relevance.

An exploratory paired cluster bootstrap uses positive source/duplicate components
as units, preserving translations together: 11 clusters, seed 20261020, 5,000
resamples. Percentile 95% intervals for logistic-minus-hybrid differences are:

| Metric | Observed difference | Exploratory interval |
|---|---:|---:|
| MRR@10 | +0.1063 | [+0.0028, +0.1806] |
| nDCG@10 | +0.0556 | [-0.0395, +0.1279] |
| Recall@50 | -0.0726 | [-0.1321, 0.0000] |

For each resample, sample 11 component IDs with replacement and divide the sum
of their query-level paired differences by their combined query count. This
retains the reported query-macro weighting. Intervals are descriptive with this
small, reused sample; nDCG does not establish a reliable population improvement.

## Production execution and resources

Two complete 180-query runs exercised `search_collection` with the learned model,
with zero query errors in both. Every candidate and original channel ranking
matches the earlier production baseline, and every learned ranking and per-query
quality record matches replay exactly. After correcting the report renderer to
include logistic rows and distinguish reranked latency, the full suite was repeated
against the final source snapshot. No ranking behavior changed in that correction.

The final run uses macOS arm64, two inference threads, the same pinned SigLIP2/E5
artifacts, and the published Tesseract generation. Training, evaluation, and manual
search verification ran with macOS outbound networking denied. Environment,
dependency versions, model registry, source hashes, and generation are recorded
in the structured report and run configuration.

| Measurement | Final result |
|---|---:|
| Model JSON | 4,931 bytes |
| Numeric parameter arrays | 336 bytes |
| Replay reranking median, per-query median of 10 calls | 0.132 ms |
| Replay reranking P95 of those per-query medians | 0.146 ms |
| Final training and replay time | 0.363 s |
| Sampled training/replay process-tree peak | 104.7 MiB |
| Full reranked request median | 1.544 s |
| Full reranked request P95 / P99 | 1.650 / 2.027 s |
| Sampled full-search process-tree peak | 1,146.6 MiB |

Replay excludes data loading, query encoding, retrieval, and file verification.
The full request includes those stages, model loading, tracing, reranking, and
thumbnail checks. Channels share a request; there is no independent simultaneous
hybrid-only latency estimate. Historical host conditions differ, so these numbers
do not establish an end-to-end speedup. RSS sampling can miss short spikes and
double-count shared pages. Native Windows/Linux inference, an 8 GB machine, and
warm resident-service latency were not measured.

## Regressions and diagnosis

The following known-positive images remain in the candidate union. Their demotion
is introduced by learned ranking, not a new decoding, OCR, or retrieval failure:

| Query | Representative known positive | Hybrid rank | Logistic rank |
|---|---|---:|---:|
| `operations-reports-fr` | `documents/synthetic/report-063.tif` | 1 | 30 |
| `operations-reports-ar` | `documents/synthetic/report-062.webp` | 5 | 20 |
| `screenshots-00-fr` | `screenshots/JabRef-jabref/0df3ea4eb7-panel-step.png` | 23 | 92 |
| `screenshots-07-fr` | `screenshots/jupyterlab-jupyterlab/de83cbb4e2-4.5-debugger-console.png` | 3 | 57 |

Original report and screenshot images were inspected. Their recovered candidates
carry 290–656 OCR characters; required content is present, and the source candidate
records are unchanged. The report and both screenshots receive negative standardized
contributions from the lexical-presence feature and BM25 magnitude. For example,
the French report receives approximately -0.707 from lexical presence and -0.460
from BM25, despite lexical rank three. The fitted weights reward missing lexical
evidence and penalize raw BM25 magnitude, while strongly rewarding reciprocal
lexical rank. These are correlated signals learned from this narrow, imbalanced
candidate sample; coefficient signs alone are not independent causal effects.

Validation contains photos and diagrams, with no screenshot or document source
families. It therefore cannot reveal this text-heavy regression during model
selection. Multi-positive workflow queries also lose nDCG when some variants fall
below the top 10, even when first-positive MRR stays high. Separately, positives
absent from the original union remain unreachable by either reranker.

No post-test feature or label adjustment was used to conceal these results. The
next quality step is independent pooled review and broader source coverage in a
new validation/test version, followed by predeclared feature ablations or ranking
objectives. A neural comparison should use the same reviewed candidates and a new
selection protocol rather than optimize against these inspected test failures.

## Preserved evidence and validation

| Ignored local artifact | Purpose |
|---|---|
| `runs/logistic-reranker-v1/final-training/` | Selected model, trials, labels, replay scores and source snapshots |
| `runs/logistic-reranker-v1/final-training-repeat/` | Independent identical-model reproduction |
| `runs/logistic-reranker-v1/final-production/` | Final 180-query production run, model copy, SQLite and JSONL records |
| `runs/logistic-reranker-v1/production-evaluation/` | Earlier complete production run, retained |
| `runs/logistic-reranker-v1/failure-diagnostics.json` | Per-positive ranks, raw features and logit contributions |
| `runs/logistic-reranker-v1/validation-final.json` | Candidate/ranking equality, holdout preservation, resume and identity checks |

The final run retains 23,122 candidate records, including 21,454 unjudged pairs.
Its train/validation/test exports contain 1,352/54/262 explicitly labeled pairs.
Resume preserved 180 query rows and 180 attempts without running inference again;
a changed model was rejected. Real personal-index CLI searches succeeded with and
without the flag. Invalid schemas/parameters, label leakage, train-only scaling,
solver convergence, deterministic fitting, display-limit stability, default hybrid
behavior, model compatibility, and Windows text handling have regression coverage.
All 193 tests, Black, Flake8, Ruff, and mypy checks passed; static type checks also
passed with Windows and Linux platform settings. Native inference on those systems
remains an uncompleted hardware/platform validation task.
