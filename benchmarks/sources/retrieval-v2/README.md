# Reranker scope and feature study

This expansion adds independently authored, legally reusable software screenshots
and document scans to the original 2,000-image collection. It addresses the first
logistic study's lack of document and screenshot validation targets. Tesseract,
the encoders, candidate budgets, and default hybrid ranking remain unchanged.

The frozen corpus has **2,052 unique images**: the original 2,000 plus 27 real
software screenshots and 25 document scans. The review covers 62 inspected images
(including 10 existing photo controls), 59 new intent families and 177 new query
translations. Combined with the original queries, study roles are 162 training,
60 fresh validation, 63 fresh test, and 72 earlier diagnostic queries.

`acquisition-plan.json` preserves all 55 originally selected new sources;
`acquisition-limits.json` records 52 acquired and three deferred screenshots after
HTTP 429. One transfer was rate-limited; the next two were not attempted during
the server's pause. The inventory was capped before any new retrieval outcomes,
without replacing difficult sources. Fresh validation has Arabic/French screenshot
interfaces but no English interface source. All three query languages are present
in every fresh fold. English screenshots remain represented in training and test.

`protocol.json` was written before new retrieval outcomes. It fixes the five
feature profiles, two scopes, four regularization values, validation guards,
tie-breaking, and fresh-test policy. Earlier validation/test queries are retained
as **diagnostic** queries and cannot select a new model. Earlier training labels,
including the separate training-only visual review, remain training labels.

The source catalog binds each file to its Commons description page, author,
license, original upload timestamp and SHA-1, download transformation and final
content hash. Commons previews are deliberately smaller than some originals;
that transformation is part of the frozen corpus. Do not substitute a different
resolution without making a new dataset version.

Software projects, document works and detected close visual copies stay in one
fold. Photo controls use a fixed-seed sample of original-corpus photographs from
source components never used as earlier positive targets. Those photographs were
already indexed: they are fresh intents on previously untargeted sources, not an
entirely pristine external image corpus. Source folds are preserved.

Review is by an agent, without independent human or native-speaker validation.
The frozen review records observed content and English/French/Arabic intents.
Only distinct inspected required subjects receive negatives; same-work variants
and declared ambiguities remain unjudged. Judgments are not exhaustive over the
corpus. Unknown images never become negative training labels.

Restore the original corpus first using the retrieval-v1 instructions, then:

```sh
uv run --no-sync python -m benchmarks.datasets.expansion \
  --base data/retrieval-2000-v1 --output data/retrieval-v2
```

Downloads are explicit and bounded. Cached files are verified. A server's HTTP
429 pause is persisted in `download-pause.json`; wait until it expires before
resuming the same command. An incomplete acquisition writes
`acquisition-errors.json` and fails without writing a partial manifest. Copies
preserve the legacy source signatures; hard links are deliberately avoided.

Rebuild the bound review artifact using:

```sh
uv run --no-sync python -m benchmarks.datasets.expansion_judgments \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --legacy-corpus benchmarks/sources/retrieval-v1/corpus.json \
  --legacy-judgments benchmarks/sources/retrieval-v1/judgments.json \
  --training-review benchmarks/sources/retrieval-v1/training-review-v1.json \
  --review benchmarks/sources/retrieval-v2/review.json \
  --protocol benchmarks/sources/retrieval-v2/protocol.json \
  --output runs/rebuilt-v2-judgments.json
```

Index once and evaluate through the real production workflow:

```sh
uv run --no-sync lens index data/retrieval-v2 --collection retrieval-v2 \
  --index-dir runs/retrieval-evaluation-index --languages eng+fra+ara --log
uv run --no-sync python -m benchmarks.experiments.collections \
  --index-dir runs/retrieval-evaluation-index --collection retrieval-v2 \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --defer-test-metrics \
  --output runs/retrieval-v2/baseline
uv run --no-sync python -m benchmarks.experiments.reranker_study \
  --baseline-dir runs/retrieval-v2/baseline \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --protocol benchmarks/sources/retrieval-v2/protocol.json \
  --output runs/retrieval-v2/study
```

Optionally add `--legacy-model PATH` to compare a previously frozen model without
refitting it. Resume collection evaluation with the identical command. Training
requires a new output directory to preserve trial history. Every production query,
candidate score, label and metric is retained. The study freezes scope champions
and any guarded selection before calculating fresh-test aggregates. If no model
passes validation, `model.json` is absent and the selection is hybrid.

`--defer-test-metrics` captures fresh-test candidates but does not compute their
ranking quality or pool recall in the baseline run. The study requires this
deferred state and scores the holdout only after `selection.json` is written.
