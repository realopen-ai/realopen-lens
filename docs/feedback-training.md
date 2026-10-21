# Local feedback training

Lens can freeze reviewed gallery feedback, fit a small logistic reranker, compare it
with production hybrid search, and explicitly activate or roll back a local model.
Training never activates a model automatically. No additional ML dependency is
required; optimization uses NumPy.

## Review before training

Save text searches with **Save for feedback** enabled and **Remember query feedback**
disabled, then label results relevant or irrelevant. In the CLI, use
`lens search QUERY --save-run --no-feedback`.
Unlabeled images are never inferred negatives. This first workflow accepts baseline
hybrid searches with 50 candidates per channel and one consistent encoder precision
and ranking-source version. Image-similarity searches, mixed encoder versions and
already reranked searches are rejected as training inputs. Searches with applied
query memory and runs containing superseded labels are also rejected. Virtual-collection
feedback retains its scope in exports but requires a scope-aware evaluation workflow;
current preparation accepts full-library searches only.

Open **Learning & models** in the gallery and download a review template. The server
retains the exact feedback snapshot locally. Fill in:

- `reviewer`: who reviewed the data.
- `runs`: select saved searches and assign each a `family` and `split` (`train`,
  `validation`, or `test`). Translations and paraphrases of the same intent belong
  to the same family and split. Select only one run per normalized query.
- `image_groups`: map every selected labeled image ID to a source-document or
  duplicate-group identifier. Related page renders, crops and duplicate encodings
  belong together, even when their byte IDs differ. Groups cannot cross splits.
- `independent_holdout`: leave false unless the reviewer can attest that held-out
  queries/judgments have not informed previous tuning.

Do not manufacture extra labels or mark a previously inspected benchmark as an
untouched holdout. If the available labels do not permit all three disjoint splits
and both training classes, collect more feedback. Lens rejects invalid input.
Source and query grouping remain explicitly reviewed. Existing exact-copy groups
and confirmed near-duplicate reviews also enforce split isolation; automatic
near-duplicate discovery is not run during preparation.

Upload the reviewed JSON, choose **Prepare reviewed dataset**, then select the
prepared dataset and **Train and evaluate logistic model**. Live job progress and
Stop use the same SSE/queue system as indexing. Prepared datasets remain selectable
after restarting the server. During final evaluation the phase display follows the
actual search stages.

## CLI

The existing feedback export command also supports a CLI-only workflow:

```sh
uv run --no-sync lens feedback export runs/review/feedback.jsonl \
  --index-dir runs/personal-index
```

Create `runs/review/review.json` using this structure, replacing the placeholders
with IDs from the export and reviewed grouping decisions:

```json
{
  "version": 1,
  "reviewer": "reviewer name",
  "feedback_file": "feedback.jsonl",
  "feedback_sha256": "SHA256 of the exact exported bytes",
  "independent_holdout": false,
  "runs": {
    "saved-run-id": {"family": "invoice-number-en-fr", "split": "train"}
  },
  "image_groups": {"image-sha256": "source-document-group"}
}
```

The complete file must include train, validation and test queries with known
positives. `feedback_file` is relative to the review file. Preparation binds the
export checksum and checks the selected labels/revisions against the feedback
store; corrections or removals since export require a fresh snapshot/review.

```sh
uv run --no-sync lens training prepare runs/review/review.json \
  runs/review/dataset --index-dir runs/personal-index

uv run --no-sync lens training train runs/review/dataset/dataset.json \
  --index-dir runs/personal-index --models models

uv run --no-sync lens training list --index-dir runs/personal-index
```

Preparation refuses an existing output directory. It preserves the exact review and
feedback export with a checksum manifest; training rejects edited frozen inputs. Each training attempt creates a
new version beneath the selected index's `rerankers/` directory. CLI progress is
JSON on stderr; the result is JSON on stdout.

## Selection and evaluation

Training uses only explicitly labeled training pairs, train-only feature scaling,
equal family/query weighting, balanced classes per query, the `core` feature
profile and L2 values 0.001, 0.01, 0.1 and 1.0. Selection uses saved validation
candidates: nDCG@10, then MRR@10, then stronger regularization. Selection is written
to disk before production test queries execute. Test labels never select a model.

Final evaluation executes paired hybrid and selected-model searches against the
saved collection generations. First method alternates by query within a shared
per-query session. Measurements include cold startup; they are not a controlled
warm-throughput benchmark. Generation changes and individual query errors remain
failed records and block activation.

Reports retain known-positive Recall@50, MRR@10, nDCG@10, judged fraction, paired
latencies, per-query regressions and separate validation/test summaries. Unknown
results contribute no known gain to these diagnostic metrics but never become
negative training labels. The qrels are nonexhaustive: these scores do not prove
corpus-wide accuracy or complete candidate recall. Parent RSS after each query and summed process-tree RSS at search phase boundaries
are reported. This sampling includes workers but is not a true peak or physical-memory
measurement; shared pages may be counted more than once. Restricted process inspection
is reported as incomplete memory coverage and does not fail a valid quality evaluation.

The model preserves the hybrid top-50 set and reranks within it. It cannot repair
candidate misses. New version-3 artifacts bind INT8 or FP32 precision explicitly;
legacy artifacts retain their INT8-only behavior. Encoder checks and the original
50/channel budget remain enforced.

Artifacts include `dataset.json`, the reviewed input snapshots/checksums, `run.json`, individual trial JSON models,
`selection.json`, selected `model.json`, `queries.jsonl` and `evaluation.json`.
They retain input hashes, optimizer/runtime source hashes, environment versions,
selection settings and individual results. Failed/cancelled attempts remain
inspectable and cannot be activated. Interrupted training starts a new attempt;
there is no optimizer resume. Local training artifacts contain queries, labels and
candidate features; they are not uploaded. Downloaded review context may also include local source paths.

## Explicit activation and rollback

Replace `MODEL_VERSION` with a completed version from `training list`:

```sh
uv run --no-sync lens training activate personal MODEL_VERSION \
  --index-dir runs/personal-index --acknowledge-experimental

uv run --no-sync lens training disable personal \
  --index-dir runs/personal-index

uv run --no-sync lens training activate personal PREVIOUS_VERSION \
  --index-dir runs/personal-index --rollback --acknowledge-experimental
```

Diagnostic holdouts or measured query regressions require explicit experimental
acknowledgment. Activation always requires an explicit request, even when the
report has no observed regression. Rollback accepts a previously activated version
for that collection and rechecks compatibility. Model/evaluation checksums and
ranking/training implementation identity are checked; modified or incompatible
versions fail explicitly rather than silently falling back. A new evaluation is
required after implementation changes.

Activation affects subsequent hybrid searches through the production search API,
CLI and gallery. Visual, lexical, semantic and image-similarity searches retain
their existing behavior. Explicit `--reranker` overrides the managed selection.
Python callers can pass `use_active_reranker=False` for baseline comparisons. The
gallery displays the ranking kind/artifact hash used for completed text searches.
Disabling restores hybrid ranking and retains model files and activation history.

Removing a feedback label does not rewrite frozen datasets or trained models.
Disable affected models and prepare a fresh reviewed snapshot before retraining.
No automatic learning or neural reranker is included.

## Audit real feedback before preparing a dataset

```sh
uv run --no-sync lens training audit --index-dir runs/personal-index-fp32
```

The read-only audit reports labeled runs, eligibility reasons, positive-only runs
and exact provenance cohorts. Cohorts separate encoder compatibility, recorded
ranking-source versions and duplicate presentation policies. Old generations,
scoped/image/already-reranked searches and runs without positives are identified
explicitly. An audit does not assign query families, choose splits or attest that a
holdout is independent. Do not rewrite old source hashes to combine cohorts.

The gallery's downloaded review template includes this audit context for each
source represented in the export. Select a compatible cohort, then complete the
source-document and query-family review. Positive-only queries are usable, but
provide limited information about false positives. Unjudged results remain unknown.

When preparing a new dataset, current exact pixel groups and **confirmed**
near-duplicate pairs constrain split isolation automatically. Connected links,
including paths through unlabeled images, must stay in one split. Dismissed and
unreviewed suggestions do not add constraints. These links do not copy relevance
labels, hide search results or assert that related images have the same relevance.
Manual source-document, crop/page and query-family grouping is still required.

The dataset freezes the constraint components and their review/snapshot evidence.
Training rechecks them alongside the existing review and feedback checksums. New
review decisions do not rewrite old datasets or models; prepare a fresh dataset
when those decisions affect isolation.

For a read-only source snapshot experiment, the benchmark adapter can copy the
published index and feedback into a new output directory. An optional duplicate
index must match the same published source snapshot:

```sh
uv run --no-sync python -m benchmarks.experiments.reviewed_feedback \
  --source-index runs/personal-index-fp32 \
  --output runs/reviewed-feedback-new
```

Without `--review`, this saves the snapshot and audit only. Add `--review PATH` to
prepare the supplied plan and run the existing logistic selection/evaluation, or
`--duplicate-index PATH` to import matching duplicate-review context from another
snapshot copy. JSON progress goes to stderr. It retains audit, freeze, selection,
per-query evaluation and error artifacts; it never activates a model.

See the [first reviewed-feedback diagnostic](benchmarks/2026-10-21-reviewed-feedback.md)
for measured results and the limits of a small, sparsely judged cohort.
