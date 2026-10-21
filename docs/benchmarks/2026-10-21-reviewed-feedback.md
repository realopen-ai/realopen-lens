# Reviewed-feedback diagnostic — 2026-10-21

A small real-feedback experiment completed through the existing logistic selection
and production evaluation workflow. It provides no evidence to promote the model:
known-judgment test quality tied hybrid, and validation quality regressed. Hybrid
remains the default. Raw queries, judgments, source paths and trained artifacts
remain local and ignored by Git; this report contains aggregate technical results.

## Audit and frozen input

The audit found 15 labeled searches with 143 explicit judgments (64 relevant and
79 irrelevant), spanning four recorded ranking-source cohorts. The largest cohort
contained 10 queries and 60 labels. The other five queries/83 labels were preserved
and excluded from this experiment; source hashes were not rewritten to combine
versions. No new relevance labels or inferred negatives were introduced.

Grouping used source-family/derived-parent provenance, byte identities, exact pixel
groups and recorded confirmed near-duplicate decisions. The imported duplicate
context contained six exact groups and ten confirmed suggestions. Dismissed
suggestions did not connect images. Query families were assigned before fitting.
A shared source family in an initial split draft was rejected; the related query
was moved wholly into training. Known duplicate constraints are now frozen and
checked automatically, including connections through unlabeled identities.

| Split | Queries | Relevant labels | Irrelevant labels |
| --- | ---: | ---: | ---: |
| Training | 6 | 11 | 35 |
| Validation | 2 | 2 | 0 |
| Test | 2 | 3 | 9 |

The cohort includes English, French and Arabic queries. This is a **diagnostic**
split with sparse, nonexhaustive judgments, not an independent holdout or a language
coverage study. The positive-only validation queries provide especially limited
model-selection evidence. Six training queries do not support a generalization
claim.

## Execution and results

The existing core feature profile, train-only normalization and fixed L2 grid
(0.001, 0.01, 0.1, 1.0) were used unchanged. Validation selected L2 0.01 before test
execution. No hyperparameters were changed after inspecting test results.
Production paired searches used an isolated 2,227-image FP32 SigLIP2 index with
INT8 E5 and two inference threads. The original index and feedback were not edited,
and no model was activated.

| Split / metric | Hybrid | Logistic |
| --- | ---: | ---: |
| Validation known-positive Recall@50 | 1.0000 | 1.0000 |
| Validation MRR@10 | 1.0000 | 0.6667 |
| Validation nDCG@10 | 1.0000 | 0.7500 |
| Test known-positive Recall@50 | 1.0000 | 1.0000 |
| Test MRR@10 | 0.5000 | 0.5000 |
| Test nDCG@10 | 0.6622 | 0.6622 |
| Test judged fraction at 10 | 0.3000 | 0.2000 |

All four held-out production evaluations completed without error. Only 20% of
hybrid top-ten positions across validation/test had judgments, compared with 15%
for logistic. Unknown results receive no known gain in these diagnostic metrics;
they never become negative training examples. Recall here measures recovery of
already known positives, not exhaustive corpus recall.

The initial experiment and two subsequent isolated reproductions selected the same
L2 and reproduced the same split-level quality metrics. The final reproduction
used the corrected evaluation order described below. Individual hybrid/logistic
rankings, candidate traces, labels, timings and failures are retained locally.

## Diagnosis

The validation regression moved a known form reference from rank 1 to rank 3.
Inspection showed a generic French form-wizard screenshot promoted above the
identifier-specific document. The target was present with valid OCR and stronger
semantic similarity; this was a learned-ranking regression, not a candidate miss.
A separate replay of all ten selected queries found all 500 saved candidate
feature rows in the current production traces, with maximum feature difference
0.0. The regression was not caused by feature drift.

The model assigned a large positive weight to raw lexical reciprocal rank and
negative weights to conservative lexical evidence and its visual interaction.
For the promoted screenshot, lexical rank contributed about +6.90 to its logit;
for the requested reference, the conservative lexical-evidence and interaction
terms contributed approximately -1.17 and -1.12. The sparse screenshot-heavy
training set did not establish a reliable identifier-specific ranking policy.
These observations explain this run; they do not establish general feature
importance or justify tuning against the observed test set.

A separate evaluation issue was found: skipping training sessions could break
alternation of which method ran first. Evaluation now enumerates only held-out
sessions, so the final four requests alternate hybrid-first/logistic-first evenly.
Progress also reports four actual evaluated queries, rather than counting training
sessions. A regression test fixes an interleaved session order to cover this case.

## Runtime and validation limits

Final per-method median latency was 1.836 seconds for hybrid and 1.658 seconds for
logistic across four paired queries. Each method ran first twice; timings mix cold
startup and warm requests. The sample is too small for a throughput or speedup claim.
Process-tree inspection was incomplete for all four queries, so retained RSS samples
do not characterize model peak memory or validate 8 GB hardware. Native execution
was macOS arm64 with Python 3.12.11; Windows/Linux inference remains unverified.

The Python suite passed 499 tests with one optional OCR skip. Black checked 186
files; Flake8, Ruff and mypy native/win32/linux checks passed for 150 source files.
API tests cover the audit context in downloaded review templates. Existing live
training progress and cancellation are reused; the experiment adapter emitted 107
JSON progress records in the final reproduction.

## Retained artifacts and next work

Ignored evidence lives under `runs/reviewed-feedback-v1/`: initial audit/export,
source grouping review, frozen input, initial model/evaluation, `validated/` and
`final/` isolated reproductions, per-query feature/diagnosis records and progress
logs. The reusable runner is `benchmarks.experiments.reviewed_feedback`; usage is
in the [training guide](../feedback-training.md).

Keep hybrid active. Before another model-selection attempt, collect more judgments
under one stable implementation and duplicate policy, especially explicit hard
negatives and independent source/query families. A broader frozen validation set
is needed. Older cohorts can support separate diagnostic checks or a separately audited
replay workflow; they remain intact. Confirmed duplicate reviews help prevent split
leakage but do not train the detector or provide relevance labels. Neural reranking
and automatic promotion are outside this experiment.

## Query-memory follow-up

Immediate exact-query feedback is now separate from learned ranking. Text searches
can prioritize explicit positives and hide explicit negatives, while baseline
training/evaluation disables the memory layer. This is deterministic personalization,
not an improvement in the model's ability to generalize.

On an isolated copy of the 2,227-image FP32 collection, native resident search
returned 50 results before and after feedback. The selected positive moved first,
the selected negative disappeared, bounded backfill retained 50 results, and clearing
restored ordinary behavior. A separate native HTTP run verified the same search,
hidden-browser and clearing lifecycle. Original feedback payloads, judgments,
events, schema version and duplicate-review digests remained unchanged.

The same frozen four-query paired evaluation was repeated with memory disabled.
All four completed; per-split Recall@50, MRR@10, nDCG@10 and judged coverage matched
the previous run exactly. Validation nDCG@10 remains 1.00 hybrid versus 0.75 logistic;
test nDCG@10 remains 0.662178 for both. The sparse diagnostic set still does not
support model promotion or an independent quality claim. Historical experiment
outputs are preserved; follow-up records live in the ignored `runs/query-memory-v1/`.

Follow-up validation: 514 Python tests passed (one optional OCR skip), 32 desktop/mobile
browser tests passed, and Black, Flake8, Ruff, mypy (native/Windows/Linux targets),
TypeScript/Vite and Prettier passed. Light/dark responsive screenshots were inspected.
Native execution remains macOS-only; Windows/Linux type targets do not establish
native inference or 8 GB hardware support.
