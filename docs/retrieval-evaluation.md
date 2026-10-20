# Saved-collection retrieval evaluation

For fresh document/screenshot source groups, diagnostic-only legacy holdouts and
guarded reranker selection, see the [retrieval-v2 protocol and commands](../benchmarks/sources/retrieval-v2/README.md)
and [measured scope study](benchmarks/2026-10-20-reranker-scope-study.md).

This benchmark evaluates the same published generations and search implementation
used by `lens index` and `lens search`. It does not rebuild a separate in-memory
index. Tesseract is the indexing OCR backend. The optional
[logistic reranker guide](logistic-reranker.md) covers training and production comparison.

## Prepare and freeze the corpus

Install development dependencies and a current regular application build as in
[the benchmark guide](benchmark.md#setup). Local pinned models and Tesseract's
English, French, and Arabic packs must already be installed.

The scale builder copies valid distinct images from the earlier mixed benchmark,
then explicitly downloads additional licensed COCO photographs. It rejects exact
encoded and production-normalized pixel duplicates. It retains source provenance, per-image
licenses, dimensions, categories, and both content hashes. The resulting corpus
is intentionally photo-heavy; its size does not establish balanced retrieval quality.

To restore the frozen collection (image bytes stay ignored by Git):

```sh
uv run --no-sync python -m benchmarks.datasets.scale \
  --base data/benchmark-500-v2 \
  --output data/retrieval-2000-v1 \
  --lock benchmarks/sources/retrieval-v1/corpus.json
```

The base dataset must be available; its generated/derived images cannot all be
restored from public URLs. Its preparation is documented in the
[large benchmark guide](large-benchmark.md). The frozen lock checks every restored
image. Never silently substitute a newer PDF, image, or synthetic render for a
checksum mismatch. COCO downloads use up to four workers with bounded batches,
size limits, timeouts, and retries. Rerunning reuses already downloaded files.
Without `--lock`, the builder creates a **new** seeded corpus; it does not imply
that existing judgments apply to it.

## Index and evaluate

```sh
uv run --no-sync lens index data/retrieval-2000-v1 \
  --collection retrieval-evaluation \
  --index-dir runs/retrieval-evaluation-index --log

uv run --no-sync python -m benchmarks.experiments.collections \
  --collection retrieval-evaluation \
  --index-dir runs/retrieval-evaluation-index \
  --corpus benchmarks/sources/retrieval-v1/corpus.json \
  --judgments benchmarks/sources/retrieval-v1/judgments.json \
  --output runs/retrieval-baseline-v1
```

The evaluator validates all source hashes and signatures before and after the run,
and source signatures and the active generation between queries. The saved
collection must contain exactly the frozen image identities. Path counts and
unique searchable image counts are reported separately. Missing, stale, or extra
images cause an explicit failure rather than changing the denominator.

Queries run sequentially with two inference threads by default. Production query
encoders run in isolated subprocesses with their existing deadlines. Each query
failure is retained and subsequent queries continue. A run with failures exits
nonzero. Successful execution alone is not a relevance verdict.

Rerun the same command to resume. `--retry-errors` retries failed queries while
preserving their previous attempts. Changing the generation, source files,
judgments, code, model directory, dependencies, or settings requires a new output
directory. A stale installed application is rejected; reinstall after editing it.
Completed records and exports are reused without inference on resume.

## Judgment protocol and split boundaries

The initial intents were authored from visual inspection before the first run.
Revision 2 subsequently corrects one report intent in three languages after a
full-resolution source audit: its section labels are numeric, rather than A–H.
Original inputs and the initial run remain retained; only corrected judgments and
the complete revision-2 repeat are authoritative. Grades and ranking code are
unchanged. The affected test intent is not an untouched holdout. English, French, and Arabic translations of an intent share a
query family and identical image judgments. The provenance explicitly identifies
agent review, without independent human or native-speaker verification. The set
is a starter evaluation, not an exhaustive or independently validated gold standard.
It contains 180 queries across 60 intent families. Twelve freshly sampled photo
targets provide 36 queries reported separately from the reused benchmark targets.
Legacy targets were previously used in pipeline development, so their test split
is not an untouched external test of the existing hybrid heuristics.

See [the frozen input notes](../benchmarks/sources/retrieval-v1/README.md) for exact
split counts and category gaps, including missing validation screenshots/documents.

Documents are grouped conservatively by source family, including repository and
synthetic-template families. Derived images retain their source family. Transitive
dHash distance of at most four joins additional likely duplicates. This heuristic
can over-group unrelated simple layouts and can miss large transformations; it is
a leakage guard, not a complete visual-duplicate detector. A seeded component hash
assigns train/validation/test at approximately 60/20/20. Components are indivisible,
so category proportions and query counts need not match those percentages.

The validator rejects crossing source documents, duplicate groups, pixel identities,
query families, or explicitly judged query/image pairs between splits. It also
rejects duplicate query identities, inconsistent corpus hashes, missing positives,
unreviewed queries, and invalid relevance grades. Near-duplicate or source groups
may contain different intents but remain in one split.

Judgments use grades 2 (directly relevant), 1 (partially relevant), and 0 (explicitly
irrelevant). The starter set uses specific known positives and distinct reviewed
counterexamples. Other candidates remain **unjudged**, including possible relevant
duplicates or related photographs. No missing label is converted into a negative.
Review additional pooled candidates and freeze a new judgment version before
claiming exhaustive recall or using them for supervised training.

All splits search the same full collection. Training exports contain only reviewed
pairs whose query and document belong to the same split. Unjudged and cross-split
candidates remain in the diagnostic export but are excluded from training files.
Do not concatenate the three training files or randomly split their rows:
translations, documents, and duplicate groups are correlated.

## Metrics and artifacts

The optional production hybrid trace exposes the same candidate union and scores
that determine search ordering. It retains all candidates (up to 150 at the default
50-per-channel budget), plus each configured channel ranking. Rankings match normal search;
tracing does not add relevant items to the candidate set.

The [candidate retrieval guide](candidate-retrieval.md) documents experimental
per-channel depths and their inclusion in evaluation resume identity. Existing
reranker models continue to require the original 50-per-channel budget.

Aggregates are query-macro averages, not total positive hits divided by all
positive judgments. Translations contribute separate query measurements, while
source/duplicate components determine independent sample size.

- Candidate Recall@50 is the fraction of reviewed positives in each channel's
  ranked first 50. Full-union recall is reported separately; a union of 150 is
  never presented as Recall@50.
- MRR@10 uses the first reviewed positive. nDCG@10 uses graded gain `2^grade - 1`
  and logarithmic discount, against the ideal order of reviewed positives.
- Unjudged items contribute zero gain to these **known-judgment metrics** but are
  never asserted irrelevant. Judged fraction at 10 exposes incomplete coverage.
- Hybrid request latency includes source validation, integrity checks, query
  subprocess/model loading, ranking, tracing, and thumbnail checks. It is not a
  warm resident-service measurement. Channel quality shares one hybrid request;
  separate channel latency is not inferred.
- Memory is sampled coordinator-plus-child RSS with a 20 ms target wait, including
  query model loading. Collection and scheduling delays increase the actual interval. Shared pages may count twice; brief spikes may be missed. Results from
  a 16 GB Mac do not establish 8 GB or native Windows/Linux support.

Each output directory retains:

| Artifact | Contents |
|---|---|
| `config.json` | Generation, checksums, signatures, code/environment identity and settings |
| `corpus.json`, `judgments.json` | Exact evaluation inputs |
| `engine-source/`, `harness-source/` | Source snapshots for historical reproduction |
| `results.sqlite` | Atomic latest per-query records plus every attempt |
| `queries.jsonl`, `attempts.jsonl` | Rankings, metrics, candidate traces, timings, memory, errors |
| `candidates.jsonl` | All candidates with raw features and nullable reviewed labels |
| `training-{train,validation,test}.jsonl` | Only explicitly labeled, same-split candidate pairs |
| `summary.json`, `report.md` | Overall and split/language/intent-category metrics |
| `resources*.json` | Sampled process-tree memory evidence |

Raw features include visual and maximum-passage semantic cosine, FTS5 BM25
(**lower is better**), channel ranks, lexical evidence, semantic strength, hybrid
score, OCR length, dimensions, and processing status. Missing scores/ranks are
null. Grade, label, source identity, category, split, and query family stay outside
the `features` object. Training must use an explicit feature allowlist, handle
missing values, and fit transforms on training data only.

Use validation families to select logistic-regression regularization first. Then
compare a small neural reranker using the same frozen candidate pool. Candidate
misses cannot be repaired by reranking. Reserve test families for a fixed final
comparison against the existing hybrid baseline; do not tune using their reports.

## Saved collections with additional images

Exact corpus membership remains the default. Explicit `--corpus-prefix` and
`--allow-extra-images` options support larger saved collections while retaining
unreviewed distractors, verifying all reference hashes, and excluding unknown
images from training labels. The [saved precision evaluation guide](saved-precision-evaluation.md)
documents matched INT8/FP32 controls and full production query comparisons.
