# Experimental logistic reranker

The logistic reranker learns how to combine existing candidate scores. It is
local, CPU-only, explicitly selected, and leaves hybrid as the default. It does
not change embeddings or require reindexing a compatible collection. The
[initial study](benchmarks/2026-10-20-logistic-reranker.md) improved average
known-judgment ranking metrics but regressed reports and some French screenshots.
It is **experimental**, not a promoted replacement for hybrid.

## Train from a production baseline

Prepare the frozen corpus, Tesseract index, and completed saved-collection baseline
as described in [retrieval evaluation](retrieval-evaluation.md). These examples
use its `runs/retrieval-baseline-v1` output. Training and search download nothing.
A fresh checkout must first restore the ignored images and models.

After editing application source, rebuild an installed regular wheel if necessary:

```sh
uv sync --locked --no-editable --reinstall-package realopen-lens
```

Keep any optional extras you already use when synchronizing. The logistic baseline
adds no dependencies: its bounded optimizer uses existing NumPy. Training stays in
`benchmarks/`; feature extraction and inference live in `src/realopen_lens/learning/`.

```sh
uv run --no-sync python -m benchmarks.experiments.logistic \
  --baseline-dir runs/retrieval-baseline-v1 \
  --corpus benchmarks/sources/retrieval-v1/corpus.json \
  --judgments benchmarks/sources/retrieval-v1/judgments.json \
  --review benchmarks/sources/retrieval-v1/training-review-v1.json \
  --output runs/logistic-training-v1
```

Training is short and deterministic. Use a new output directory for every experiment;
existing outputs are refused to preserve previous trials. The corpus, canonical
judgment hashes, trace schema, complete query inventory, exact query references,
and candidate identities are checked before fitting.

The frozen additional visual review covers 212 training family/image pairs:
165 explicit negatives, four direct matches, and 43 ambiguous pairs retained as
unjudged. Decisions apply to all three translations, adding 507 judgments.
Only 429 new labeled pairs occur in the saved pools, giving 1,352 training pairs:
116 positive and 1,236 negative. Validation/test judgments and group splits are unchanged.

This is **agent review**, without independent human or native-speaker review.
Selection policy, observations, reasons, input checksums, and provenance are in
the review JSON. It never turns an unknown label into a negative. A different
corpus or revised canonical judgments require a new bound review.

## Fitting and selection

The version-1 feature allowlist contains 14 values:

- Visual/semantic cosines, lexical evidence, semantic strength, and hybrid score.
- Missing semantic/lexical indicators and `log1p(-BM25)` (FTS5 BM25 is lower-is-better).
- Reciprocal visual, semantic, and lexical top-50 ranks, with zero for missing ranks.
- `log1p(OCR character count)` and visual/lexical and visual/semantic interactions.

IDs, paths, categories, splits, languages, labels, and query families are excluded
from the feature vector. Scaling is fitted on training pairs only; constant columns
use scale one. Family and translation weights prevent example counts from deciding
relative influence. Classes are balanced within each query. Unknown and cross-split
pairs are excluded from training.

The deterministic Newton optimizer minimizes weighted binary log loss plus
`L2 / 2 * ||weights||²`, with an unpenalized intercept, stable log-sum-exp,
backtracking, a 100-step bound, and an explicit convergence check. It is a
pointwise logistic baseline, not pairwise or listwise learning to rank.

The fixed L2 grid is `0.001, 0.01, 0.1, 1`. Selection uses validation nDCG@10,
then validation MRR@10, then stronger regularization. The selected model and its
checksum are frozen before evaluating test quality. The initial study selected
`0.01`; no parameters were changed in response to the inspected test results.
Scores are logits for ordering, without a probability-calibration claim.

Outputs preserve trial models, `selection.json`, `model.json`, reviewed judgments,
labeled training rows, input/source/environment identities, source snapshots,
per-query scores/rankings/metrics, and resource/timing summaries. The deployed JSON
contains numeric arrays, schemas, encoder compatibility, and provenance hashes;
it excludes the full training inventory and private source paths. No pickle or ML
framework is loaded.

## Try the model

```sh
uv run --no-sync lens search "pizza" \
  --index-dir runs/personal-index \
  --reranker runs/logistic-training-v1/model.json
```

Omit `--reranker` for normal hybrid ranking. The flag works only with hybrid mode.
It explicitly rejects incompatible encoders/preprocessing, malformed schemas,
invalid scales, nonfinite coefficients, and model files over 1 MiB.

Learned candidate retrieval uses 50 per channel, with a union of at most 150,
independent of `--limit`. Version-1 models and version-2 `union` models reorder that
union before applying the display limit. They cannot recover positives outside
the pool. **Full-pool recall stays
constant; recall within the reordered first 50 can improve or regress.**

Version-2 `hybrid50` models reorder only the baseline hybrid top 50 and append the
baseline tail unchanged. This preserves top-50 membership, including when the
display limit changes. It guarantees the same known-positive Recall@50 as hybrid;
it does not guarantee better MRR or nDCG.

## Fresh-source scope and feature study

The [retrieval-v2 protocol and commands](../benchmarks/sources/retrieval-v2/README.md)
separate the earlier held-out families as diagnostic queries. They add fresh
document and screenshot source groups, previously untargeted photo controls,
and English/French/Arabic intents frozen before retrieval. Review is by the same
agent, without independent human or native-speaker validation.

The predefined comparison fits 40 models: `union` and `hybrid50`, four L2 values,
and five feature profiles. Profiles exclude BM25 magnitude, OCR length, or missing
evidence indicators individually; `core` excludes all four of those feature
columns. Excluded columns have zero coefficients and neutral scaling. The fixed
14-column trace schema is unchanged, and labels/identities never enter features.

Only training pairs fit scaling and coefficients. Selection uses fresh validation
and requires no Recall@50 loss overall or within any query language or image kind,
no MRR@10 loss, and strictly better nDCG@10. Among passing models it uses nDCG,
MRR, stronger L2, then stable profile/scope names. If no model passes, hybrid is
the selected policy and no `model.json` is written.

Baseline evaluation uses `--defer-test-metrics`: it captures fresh-test candidates
without scoring their quality. The study refuses previously scored fresh-test
records. It freezes each scope's validation champion and any guarded selection,
then evaluates the fresh holdout. Research champions remain available even
when they fail the selection guard. The old fixed model can be compared with
`--legacy-model`, without refitting it. Default production search remains hybrid.

After running the study, try its protected-scope artifact explicitly:

```sh
uv run --no-sync lens search "pizza" --index-dir runs/personal-index \
  --reranker runs/retrieval-v2/study/champion-hybrid50.json
```

`model.json`, when present, is the unchanged validation-selected artifact; it can
use `union` and therefore does not carry the membership guarantee. See the
[fresh-source comparison](benchmarks/2026-10-20-reranker-scope-study.md) for the
measured tradeoffs and unresolved failures.

## Evaluate through production search

```sh
uv run --no-sync python -m benchmarks.experiments.collections \
  --collection retrieval-evaluation \
  --index-dir runs/retrieval-evaluation-index \
  --corpus benchmarks/sources/retrieval-v1/corpus.json \
  --judgments runs/logistic-training-v1/judgments.json \
  --reranker runs/logistic-training-v1/model.json \
  --output runs/logistic-production-v1
```

Each real search request retains both hybrid and logistic rankings. This gives
paired quality on shared candidates, not independent channel latency estimates.
Full request time is `reranked_end_to_end_seconds`. Replay timing isolates ranking
cost and must not replace full request or warm resident-service measurements.

Rerun the identical evaluation command to resume without repeating completed queries.
`--retry-errors` retains previous failed attempts. A different model checksum,
collection, implementation, or evaluation input requires a new output directory.
The run copies its model and retains all individual query results.

The initial relevance set is incomplete and small. Its validation had no document
or screenshot families. Its test was previously reported for hybrid and contains a corrected report
intent: it is held out from this fit, not an untouched external test. Independent
pooled review and broader source-family coverage remain necessary before promotion
or neural-reranker selection; the fresh-source study improves coverage without
establishing production-wide accuracy.
