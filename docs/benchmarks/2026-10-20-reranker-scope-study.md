# Fresh-source reranker scope study

Date: 2026-10-20. **Experimental; hybrid remains the default.**

The expanded-source experiment improves known-judgment ranking on fresh sources. Overall test Recall@50 is unchanged, but a photo loss is offset by a screenshot gain. The top-50-only scope mechanically avoids that membership loss. Neither result justifies automatic promotion.

See the [reproduction guide](../../benchmarks/sources/retrieval-v2/README.md), [structured report](2026-10-20-reranker-scope-study.json), and [initial study](2026-10-20-logistic-reranker.md).

## Dataset and frozen protocol

The corpus contains 2,052 unique images: 1,657 photos, 145 screenshots, 124 documents, 79 diagrams and 47 challenging cases. It retains the original 2,000 images and adds 27 real software screenshots and 25 document scans from 32 source groups. Per-file author, license, provenance, original SHA-1, final content/pixel hashes and transformations are frozen. Exact duplicates and close visual copies are checked; this is not an exhaustive semantic duplicate detector.

All 52 acquired sources were frozen before new retrieval outcomes. The original 55-source acquisition plan and three deferred screenshot records remain auditable. Fresh validation lacks English interface screenshots, while training/test contain them. All fresh intents have English/French/Arabic translations.

Visual review covered 62 images, including ten fixed-seed previously untargeted original-corpus photo controls. The 59 new intent families add 177 queries. They are agent-reviewed, without independent human/native-speaker validation. Unknown and ambiguous images are not negative labels. The photo controls were previously indexed, so this is not an entirely pristine external corpus.

Study roles are 162 train, 60 fresh validation, 63 fresh test and 72 earlier diagnostic queries. Software projects, document works, duplicate components and translated families stay within a fold. Earlier validation/test families are excluded from selection. All canonical legacy labels are retained, including the earlier train-only review.

The predeclared protocol compares five feature profiles × two scopes × four L2 values = 40 trials. Only train fits coefficients/scaling. Validation requires no Recall@50 loss overall/per-language/per-kind, no overall MRR loss and a strict nDCG gain. Selection is frozen before fresh-test scoring; the baseline defers all 63 fresh-test quality calculations.

## Selection and ablations

Training uses 1,593 explicitly labeled pairs (166 positive) across 54 families. There are 261 new-family pairs (50 positive); the expanded pool retains 1,332 legacy pairs. No validation/test pair is fitted. Nineteen trials pass the validation guard. All profile champions favor L2=1 over the original model’s 0.01.

Validation selected `no-ocr-length`, `union`, L2=1. Its artifact is 4,905 bytes. The independently frozen `hybrid50` champion has the same profile, L2 and coefficients; only scope differs. The original model is a fixed reference. The small difference between profiles does not establish that dropping BM25 or missing indicators is beneficial, or isolate data expansion from regularization.

| Profile | Scope | L2 | Validation Recall@50 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| full | union | 1 | 0.9333 | 0.5797 | 0.6341 |
| full | hybrid50 | 1 | 0.9333 | 0.5746 | 0.6262 |
| no-bm25 | union | 1 | 0.9333 | 0.5797 | 0.6341 |
| no-bm25 | hybrid50 | 1 | 0.9333 | 0.5746 | 0.6262 |
| no-ocr-length | union | 1 | 0.9500 | 0.5839 | 0.6375 |
| no-ocr-length | hybrid50 | 1 | 0.9333 | 0.5788 | 0.6296 |
| no-missing | union | 1 | 0.9333 | 0.5809 | 0.6315 |
| no-missing | hybrid50 | 1 | 0.9333 | 0.5742 | 0.6258 |
| core | union | 1 | 0.9500 | 0.5837 | 0.6372 |
| core | hybrid50 | 1 | 0.9333 | 0.5785 | 0.6293 |

## Paired quality results

Metrics are macro averages over reviewed known judgments, not exhaustive accuracy. Fresh-test judged fraction at 10 is only 0.2460 for hybrid and 0.2556 for the selected model: most returned items remain unjudged. Unknowns contribute zero gain to these metrics but never become negative training labels. Candidates and references are identical across the rankings. Scope champions were frozen on validation; the slightly higher test nDCG of `hybrid50` was not used to change selection.

| Cohort | Ranking | Recall@50 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|
| validation (60) | hybrid | 0.9333 | 0.5399 | 0.5908 |
| validation (60) | legacy | 0.9167 | 0.5552 | 0.6053 |
| validation (60) | union | 0.9500 | 0.5839 | 0.6375 |
| validation (60) | hybrid50 | 0.9333 | 0.5788 | 0.6296 |
| test (63) | hybrid | 0.9206 | 0.5005 | 0.5506 |
| test (63) | legacy | 0.8333 | 0.5359 | 0.5686 |
| test (63) | union | 0.9206 | 0.5897 | 0.6195 |
| test (63) | hybrid50 | 0.9206 | 0.5919 | 0.6247 |
| diagnostic (72) | hybrid | 0.9111 | 0.5608 | 0.5987 |
| diagnostic (72) | legacy | 0.8931 | 0.6575 | 0.6628 |
| diagnostic (72) | union | 0.9257 | 0.6191 | 0.6440 |
| diagnostic (72) | hybrid50 | 0.9111 | 0.6211 | 0.6456 |

Fresh-test full-pool known-positive recall is 0.9524 for every reranker. All 357 queries preserve exact top-50 membership under `hybrid50`. The selected union increases screenshot recall from 0.8889 to 0.9259 but lowers photo recall from 0.8333 to 0.7500; overall 0.9206 hides that exchange. Specifically, the Arabic Inkscape palette target moves from hybrid rank 52 to union 31 while the Arabic snowboarder moves 38 to 72.

| Fresh test group | Hybrid MRR | Selected MRR | Hybrid nDCG | Selected nDCG | Hybrid Recall@50 | Selected Recall@50 |
|---|---:|---:|---:|---:|---:|---:|
| kind:documents | 0.5563 | 0.6645 | 0.6223 | 0.7137 | 1.0000 | 1.0000 |
| kind:screenshots | 0.5722 | 0.6831 | 0.6023 | 0.6812 | 0.8889 | 0.9259 |
| kind:photos | 0.2278 | 0.2300 | 0.2905 | 0.2923 | 0.8333 | 0.7500 |
| language:en | 0.6131 | 0.8243 | 0.6825 | 0.8619 | 1.0000 | 1.0000 |
| language:fr | 0.5234 | 0.5729 | 0.5750 | 0.5969 | 1.0000 | 1.0000 |
| language:ar | 0.3651 | 0.3719 | 0.3941 | 0.3997 | 0.7619 | 0.7619 |

A descriptive paired bootstrap over 14 positive source components (10,000 draws, seed 20261020; translations and same-project intents clustered) gives selected nDCG delta +0.0689, 95% interval [0.0229, 0.1182]. This is conditional on this small incomplete agent-reviewed set, not a population accuracy guarantee; the interval was not used for selection.

## Failure investigation

Original targets and competing outputs were inspected in 30 diagnostic image panels. Cached OCR payloads, target channel ranks, per-feature logit contributions and per-query rankings are retained in `runs/retrieval-v2/diagnostics/`. These post-freeze observations do not alter labels or model selection.

| Case | Observation and root cause | Action / remaining work |
|---|---|---|
| Arabic snowboarder | Hybrid 38 → union 72; hybrid50 49. Weak visual rank 37 and no OCR evidence; semantic-derived features let noisy-text candidates overtake the visual-only target. | Top-50 scope fixes membership. Union photo recall remains a real test regression; add train-only hard negatives and test constrained evidence features in a new study. |
| Arabic J’Accuse newspaper | Tesseract deadline leaves zero text/semantic evidence. Hybrid 4 → union 12 / hybrid50 11. | Degraded status and visual fallback work; dense multilingual OCR remains unresolved. No holdout-specific timeout change. |
| English KDE multi-positive | Second correct screenshot 2 → 3; a generic configuration screenshot with lexical rank 1 but gated lexical evidence 0 moves 11 → 2. | Raw channel ranks can bypass the baseline evidence gates. Top-50 membership alone cannot guarantee nDCG. |
| Arabic pizza platter | Known positive 6 → 7 under union; hybrid50 keeps rank 6. | Small ranking regression retained, not hidden by averages. |
| Three Arabic pool misses | Cafeteria, Inkscape welcome and Wikipedia home targets absent from the union. Originals/OCR inspected; full ranks recomputed with English controls. | Improve candidate retrieval before expecting a neural reranker to recover these targets. Labels remain incomplete. |
| Windows Unicode manifests | Real UTF-8 manifest fails under CP1252 or yields mojibake paths. | Explicit UTF-8 read plus Arabic filename/OCR/search regression test. |

Full-channel diagnostics and the independent OCR timeout reproduction are recorded in the structured report. No model changes were made after inspecting fresh-test results.

## Production, resources and verification

Actual collection indexing and both complete 357-query production evaluations run with OS-level networking denied, using pinned int8 SigLIP2 and multilingual E5 plus Tesseract 5.5.2 (`eng+fra+ara`, two threads). The generation contains 2,052 images, zero failed image paths and one explicitly OCR-degraded newspaper. Earlier indexing passes reuse completed stages.

The final production run follows the one-line UTF-8 storage fix. Baseline and final code hashes differ for that reason; every candidate/channel ranking and final learned output is compared against the frozen baseline/replay. The interrupted pre-fix evaluation and its SQLite records are preserved and excluded from complete-run metrics.

| Run | Queries / errors | Median seconds | P95 | P99 | Peak sampled process-tree MiB |
|---|---:|---:|---:|---:|---:|
| Hybrid baseline | 357 / 0 | 1.600 | 1.851 | 2.142 | 1147.6 |
| Selected production | 357 / 0 | 1.802 | 2.022 | 2.650 | 1155.0 |

Forty fits plus all finalist replays take 4.30s with 131.3 MiB sampled peak. Selected ranking-only median is 0.139 ms; it is not full-request latency. The sequential selected median is 12.6% higher (1.600 → 1.802s). A subsequent 12-query paired alternating probe has medians 1.747s for hybrid and 1.748s for the selected model, with median paired delta −0.026s. It does not reproduce the sequential difference as consistent reranker overhead. Wall times include encoder subprocess startup, model/artifact checks, tracing and thumbnails; background load and OS caching are not controlled. No speedup or precise overhead claim is made.

Black, Flake8, Ruff, mypy (default, Windows and Linux targets) and 216 tests pass locally. Tests cover protected membership with a production pool larger than 50, neutral ablation parameters, unchanged v1 compatibility, subgroup guards, held-out scoring deferral, dataset leakage/hash consistency, resumable rate limits, legacy file signatures and Unicode manifest search. Native Windows/Linux inference and 8 GB hardware are not certified by these macOS runs.

All per-query candidate scores/labels and attempts remain in SQLite and JSONL under `runs/retrieval-v2/{baseline,study,production-selected}`. The structured report also publishes compact individual results for all 357 queries. Exact model hashes, input hashes, dependency versions and environment details are retained.

## Remaining work

Keep hybrid as the default. Use `hybrid50` when experimenting with a hard top-50 membership guarantee; keep the union artifact as the unchanged validation-selected research result. Before neural selection, broaden independent human/native-speaker review, add train-only gated-evidence hard negatives, fill the English validation-interface gap, investigate Arabic candidate coverage and dense-page OCR, then freeze a new holdout. The current holdout is now diagnostic for future tuning.
