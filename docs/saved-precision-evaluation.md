# Evaluate a full saved collection across SigLIP2 precisions

The collection evaluator remains strict by default: the saved collection must
contain exactly the frozen corpus, at its recorded paths. Two explicit options
support evaluating a larger personal collection without discarding distractors:

- `--corpus-prefix retrieval-v2` locates corpus paths under the collection root.
- `--allow-extra-images` retains additional searchable images as unreviewed
  distractors and records their count separately from the reference corpus.

Every reference image must still be present at its expected path with the correct
hash. Superset evaluation also verifies every additional image and duplicate path,
then checks filesystem signatures and generation identity throughout the run.
Changes to the collection or inputs invalidate the run rather than silently
shrinking the search space.

Additional images receive no invented provenance, split, category, or relevance
label in candidate exports. They are excluded from training exports. The existing
training validators continue to require their exact frozen corpus and encoder
compatibility; these expanded-collection traces are evaluation evidence, not
drop-in training artifacts.

## Build a matched INT8 control

Start with a published FP32 collection using the current processing configuration:

```sh
uv run --no-sync python -m benchmarks.experiments.saved_precision \
  --index-dir runs/personal-index-fp32 --collection personal \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --corpus-prefix retrieval-v2 --allow-extra-images \
  --output runs/full-precision-local/control
```

The preparation command creates a separate INT8 collection. It requires matching
runtime/model/OCR configurations apart from SigLIP2 precision, verifies and copies
metadata/OCR/E5 cache records, and recomputes visual embeddings through production
indexing. It checks identical members, statuses, OCR text, and E5 vectors afterward.
The original FP32 index is read-only. Existing unrelated target directories are
rejected; cached controls and completed vision stages can resume on an identical
retry. Output must be outside the original index and image root.

This isolates the encoder precision change. The preparation timing is a cached
control build, not a fresh end-to-end indexing speed comparison. Source input
failures are retained in `source-input-failures.json` and indexing attempts,
separately from searchable images and query execution failures.

## Run the complete production queries

```sh
uv run --no-sync python -m benchmarks.experiments.collections \
  --index-dir runs/full-precision-local/control/int8-index --collection personal \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --corpus-prefix retrieval-v2 --allow-extra-images \
  --output runs/full-precision-local/int8

uv run --no-sync python -m benchmarks.experiments.collections \
  --index-dir runs/personal-index-fp32 --collection personal \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --corpus-prefix retrieval-v2 --allow-extra-images \
  --output runs/full-precision-local/fp32

uv run --no-sync python -m benchmarks.reporting.precision \
  --int8 runs/full-precision-local/int8 --fp32 runs/full-precision-local/fp32 \
  --output runs/full-precision-local/comparison
```

Keep the source collection and image files unchanged during evaluation. Run the
policies sequentially with the same thread count, models and candidate budgets.
Queries perform complete production search, including model integrity checks and
short-lived encoder startup. SQLite retains individual queries and attempts;
JSONL exports retain candidates, channel scores, labels, errors, latency and sampled
process-tree RSS. Summary reports include language/kind metrics and latency
distributions. Identical commands resume completed queries; `--retry-errors`
retries failed requests. Source/configuration changes require a new output directory.

The comparison rejects incomplete, unscored or mismatched runs and rechecks the
live collections' OCR/E5 controls and generation identities. It preserves paired
query metrics, known-positive ranks and nDCG regressions. Inspect
`memory_complete` and `memory_access_errors` before interpreting resource results;
restricted process access can otherwise leave only coordinator RSS visible.

The comparison checks the retained SQLite attempt history against final query
exports. Ranking-quality tables use the final successful results; first-pass
successes, failures, deadline-aware quality, latency including timeouts and every
failed attempt remain separate in `execution_history`. A successful retry never
erases the original failure. All-attempt memory sampling contributes to
`memory_complete`, including failed requests. `reporter_sha256` identifies the
postprocessor separately from the frozen collection harness in each run's config.
Changing reporting code after collection does not change the retained raw results,
but changes the harness identity for future collection/resume commands; finish all
retries before updating code, or use the retained source snapshot.

The [full saved-collection study](benchmarks/2026-10-20-full-siglip2-precision.md)
records an executed comparison, including its timeout and recovery.

## Interpreting results

Metrics use the frozen known judgments. Unjudged competitors can be relevant;
a lower known-judgment nDCG does not establish that every competing result is wrong.
The old reviewed query set is diagnostic, not a new independent holdout.
An expanded search space and updated OCR make older, smaller benchmark runs
unsuitable as controlled baselines. Evaluate the two matched policies directly.

Native platform validation and constrained-memory testing remain separate from
static checks or a successful run on a larger macOS machine.
