# Production retrieval evaluation — 2026-10-20

This baseline evaluates a saved production collection with the existing hybrid ranking. No logistic regression or neural reranker was trained, and no ranking thresholds were tuned during this study.

## Implementation

- The evaluator calls production `search_collection` against a published generation, using the same paths as `lens index` and `lens search`.
- Optional tracing retains the candidate union, channel scores/ranks, lexical evidence, semantic strength, and hybrid score without changing ranking.
- Run identities freeze generation, source signatures, corpus/judgment hashes, source code, models, dependencies, and configuration. SQLite retains each query and all retry attempts; JSONL and training exports are derived from those records.
- Unknown labels stay null. Training files contain only reviewed candidate pairs in the same query/document split.

## Corpus and judgments

The corpus has **2,000 paths and 2,000 searchable unique images**, checked by encoded SHA256 and by production-normalized RGB pixels after EXIF orientation and alpha compositing. It combines 492 valid, deduplicated images from the previous mixed corpus with 1,508 additional licensed COCO photographs. Three expected decode rejections and five identical-pixel encodings were excluded from the earlier corpus.

| Category | Images |
|---|---:|
| challenging | 47 |
| diagrams | 79 |
| documents | 99 |
| photos | 1657 |
| screenshots | 118 |

There are **180 queries, 60 intent families, and 5,952 explicit judgments** in English, French, and Arabic. The inspected targets cover 88 images; multilingual variants share judgments and remain in one split. Twelve newly sampled photo targets supply 36 fresh queries, separate from 144 legacy queries. The initial set was frozen before retrieval. A later full-resolution source audit corrected one report intent in all three languages: its section labels are numeric 1–9, rather than the mistakenly specified A–H. Revision 2 removes that unsupported constraint without changing relevance grades or ranking code. The entire 180-query evaluation was repeated against corrected frozen inputs; all 177 unchanged query traces matched the original exactly. The corrected test intent is not an untouched holdout.

| Split | Images | Queries | Intent families | Positive source/duplicate components |
|---|---:|---:|---:|---:|
| train | 1226 | 108 | 36 | 16 |
| validation | 346 | 30 | 10 | 7 |
| test | 428 | 42 | 14 | 11 |

**Review provenance:** Codex visually inspected the targets and authored/checked translations. These are agent-reviewed starter judgments, without independent human or native-speaker review. Relevance is incomplete. Related images may be relevant even when unjudged; unknown labels are never confirmed negatives.

Legacy images were used in earlier pipeline development. Their new split prevents future supervised training leakage, but does not make them an untouched external test of existing hybrid heuristics. Validation covers photos and diagrams, with no document or screenshot queries. Test text coverage includes synthetic reports and workflows. These gaps and only 34 positive components limit generalization claims, especially for a neural reranker.

The initial inputs/run remain retained for audit; only revision-2 results and exports are authoritative. The exact correction and checksum preconditions are in [the correction record](../../benchmarks/sources/retrieval-v1/corrections-v2.json). Full provenance, license metadata, hashes, and review records are in [the frozen input bundle](../../benchmarks/sources/retrieval-v1/README.md).

## Results

All **180/180 queries** completed; there were **0 execution failures**. Metrics below are query-macro averages over reviewed known positives, not exhaustive relevance. Unjudged results receive zero gain for these metrics without becoming negative training labels.

| Ranking | Known-positive Recall@50 | MRR@10 | nDCG@10 | Judged fraction@10 |
|---|---:|---:|---:|---:|
| visual | 0.8681 | 0.4860 | 0.5301 | 0.1900 |
| lexical | 0.4917 | 0.4181 | 0.4335 | 0.2159 |
| semantic | 0.3828 | 0.2791 | 0.2890 | 0.1033 |
| hybrid | 0.9378 | 0.6087 | 0.6574 | 0.2422 |

The full candidate union has mean known-positive recall **0.9506**. This union can contain up to 150 images; it is reported separately from each ranked top 50.

| Hybrid subset | Queries | Recall@50 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| split:train | 108 | 0.9537 | 0.6255 | 0.6825 |
| split:validation | 30 | 0.9667 | 0.4661 | 0.5151 |
| split:test | 42 | 0.8762 | 0.6675 | 0.6945 |
| exposure:legacy-benchmark | 144 | 0.9361 | 0.6402 | 0.6862 |
| exposure:new-scale | 36 | 0.9444 | 0.4827 | 0.5423 |
| language:en | 60 | 0.9633 | 0.7106 | 0.7540 |
| language:fr | 60 | 0.9583 | 0.5896 | 0.6488 |
| language:ar | 60 | 0.8917 | 0.5260 | 0.5695 |
| kind:photos | 72 | 0.8611 | 0.4191 | 0.4768 |
| kind:screenshots | 30 | 1.0000 | 0.7486 | 0.7772 |
| kind:documents | 39 | 0.9692 | 0.8235 | 0.8505 |
| kind:diagrams | 39 | 1.0000 | 0.6364 | 0.7057 |

## Latency and resources

Hybrid request latency: mean **3.106s**, median **1.866s**, P95 **9.792s**, P99 **10.847s**. Peak sampled query process-tree RSS was **1144.5 MiB**.

Requests include source and artifact validation, model/query subprocess startup, ranking, tracing, and thumbnail checks. File caches were not flushed. A latency investigation during the initial run observed 8.34 GiB of host swap in use, 4.07 GiB available RAM, and other CPU workloads, without a reported thermal warning. The slow requests were transient; a specific cause was not established. These are busy-host measurements, not an isolated hardware-speed baseline. Channel quality shares the hybrid call; separate channel latency is not claimed. Memory samples include descendants with a 20 ms target wait; collection and scheduling can increase the interval. Shared pages can count twice and brief spikes can be missed.

Production indexing used two incremental passes while downloads completed: first 923 images, then a refresh publishing 2000 images with 923 fully reused images. Combined reported indexing time was **1506.7s**; final refresh time was **798.6s**. Final publication had **0 failed paths** and **0 degraded images**. Final-refresh peak sampled tree RSS was **627.2 MiB**; initial-pass memory was not sampled. This is not one cold 2,000-image indexing measurement.

Environment: one 16 GiB arm64 Mac, CPUExecutionProvider, INT8 models, two inference threads, Tesseract 5.5.2. Software versions, pinned model revisions, language-pack hashes, and aggregates are retained in [the structured results](2026-10-20-retrieval-evaluation.json). There is no native Windows/Linux or 8 GiB hardware measurement.

## Candidate exports and diagnostics

| Partition | Positive candidate pairs | Negative candidate pairs |
|---|---:|---:|
| train | 104 | 819 |
| validation | 30 | 24 |
| test | 104 | 158 |

The export retains **23122 candidates**, including **21883 unjudged pairs**. Counts above describe candidates actually returned, not all input judgments. Missing positives are not injected into the pool for training.

Diagnostic groups:

- ranking: 24 queries.
- candidate_pool: 10 queries.

`candidate_pool` means at least one reviewed positive was absent from the full candidate union, including partial loss in multi-positive queries. `ranking` means the pool retained all reviewed positives but none appeared in the hybrid top 10. These are known-positive diagnostics, not definitive judgments about the relevance of unreviewed returned images.

Representative inspected cases include a dark fisheye photo lost during candidate generation, a French terminal query retained at rank 47 with zero OCR boost, and an armored-person/horse photo retained below the top ten. The fresh cow image is lost for the Arabic query; its English/French comparisons and exact target features are retained in the structured results. These are evidence-backed stage diagnoses, not guarantees that a learned ranker will resolve the cases.

Affected query IDs are in the structured results. Per-query traces, missing-positive IDs, and positive candidate features are retained in `runs/retrieval-evaluation-v1/failure-diagnostics.json`; source images and OCR analyses remain in the local corpus and production catalog. Candidate losses set a ceiling for any reranker operating on this frozen pool.

## Validation and next steps

- 173 tests pass; Black, Flake8, Ruff, and mypy checks pass. Type checks cover macOS, Windows, and Linux targets; this is not native execution on all three.
- Production indexing and evaluation ran under a macOS OS-level network deny policy. Only Tesseract was evaluated.
- A full resume preserved every query and attempt record without query inference. Incompatible run identity was rejected before creating result records.
- Twelve independent production-mode searches (three queries across four modes) matched the traced rankings exactly. Training exports passed same-split and explicit-label checks.

Train logistic regression first using only explicit training pairs and an allowlist of numeric features. Fit missing-value handling and scaling on training data, tune on validation families, and compare against the fixed hybrid ordering on the same candidate sets. Do not tune on the test results above.

Expand independent real document/screenshot families and pooled relevance review, including high-scoring hard negatives and independent language review, before relying on validation across all domains or drawing conclusions about a neural reranker. Keep candidate losses distinct from ranking errors. This branch establishes the evaluation/export baseline; learned-ranker comparisons remain future work.

## Retained evidence

- `data/retrieval-2000-v1/`: 2,000 verified image files and manifest.
- `runs/retrieval-evaluation-index/`: production catalog, immutable generation, thumbnails, OCR analyses and vectors.
- `runs/retrieval-evaluation-v1/baseline-reviewed-v2/`: every query/attempt, SQLite results, candidate and training JSONL exports, summaries, memory evidence and source snapshots.
- `runs/retrieval-evaluation-v1/`: acquisition/corpus verification, indexing logs/results, final-refresh memory evidence, full evaluation/resume logs, mismatch rejection and production-mode parity checks.

Images, models, index stores and raw run evidence stay ignored locally. The repository tracks the frozen metadata/judgments, benchmark implementation, tests, and this aggregate report. Follow [the reproduction guide](../retrieval-evaluation.md) to restore inputs and run a new local evaluation.
