# Production SigLIP2 precision validation

Date: 2026-10-20. INT8 remains the default; FP32 SigLIP2 is opt-in and E5 stays INT8.

FP32 is now exercised through production indexing, persisted collections, query workers,
and hybrid search. On this diagnostic subset, MRR@10 increases from 0.6508 to 0.8868
and nDCG@10 from 0.7152 to 0.9145, with higher cold-query latency and memory.

See the [setup guide](../siglip2-precision.md) and [structured individual results](2026-10-20-siglip2-precision.json).

## Scope and controls

The deterministic subset contains 128 images: 37 photographs, 36 documents,
30 screenshots, 20 diagrams, and 5 challenging images. Seed 20261020 selects
two English families per kind, retaining their translations and every reviewed label;
seeded distractors fill the remaining places. There are 8 families and 24 queries,
8 each in English, French, and Arabic. Original licensing/provenance and splits
remain in the subset manifest. These are previously inspected, agent-reviewed
judgments without independent human review, not a fresh holdout.

Both separate indexes contain all 128 images, zero failed files, and one degraded
newspaper with partial OCR. OCR records and E5 passage vectors are bit-identical.
Pipeline specifications differ only in SigLIP2 precision. All 128 native FP32
image vectors match the earlier cached FP32 diagnostic exactly. All 24 queries
per policy reproduce the corresponding cached visual rankings, with maximum
cosine error below 3e-8. Native query repeats preserve all 48 traces and metrics.

## Quality and cold requests

| Policy | Recall@50 | MRR@10 | nDCG@10 | Median s | P95 s | Peak tree MiB |
|---|---:|---:|---:|---:|---:|---:|
| int8 SigLIP2 | 1.0000 | 0.6508 | 0.7152 | 1.529 | 2.201 | 1043.5 |
| fp32 SigLIP2 | 1.0000 | 0.8868 | 0.9145 | 2.859 | 4.022 | 2891.5 |

Each policy completes 24/24 native requests without errors. Recall@50 is saturated
on this smaller search space; it does not supersede the earlier 2,052-image study.

| Language | INT8 MRR@10 | FP32 MRR@10 | INT8 nDCG@10 | FP32 nDCG@10 |
|---|---:|---:|---:|---:|
| en | 0.7708 | 0.8375 | 0.8273 | 0.8772 |
| fr | 0.6637 | 0.9167 | 0.7439 | 0.9375 |
| ar | 0.5179 | 0.9062 | 0.5744 | 0.9288 |

There are 0 known-judgment nDCG regressions in this subset; individual
query metrics and top-10 image IDs are retained in the structured report.

## Indexing and resource measurement

| Policy | Initial full build s | Encoder rebuild s | Rebuild peak tree MiB |
|---|---:|---:|---:|
| int8 | 265.05 | 18.13 | 906.1 |
| fp32 | 284.89 | 35.04 | 886.1 |

Initial full-build times include scanning, metadata, OCR, encoders, and publication.
Their sandboxed child-process memory samples were incomplete and are excluded.
Native resource probes clone each completed index, remove only vision/semantic
analysis records, and rebuild those stages for all 128 images, reusing OCR and
metadata. These model-stage measurements are distinct from full fresh-index memory.
Native queries and encoder rebuilds have zero memory-sampling access errors.

Requests include integrity verification and short-lived encoder startup. Runs are
sequential on one macOS CPU host with two threads; OS caches and background load
are not controlled. RSS is sampled every 20 ms and shared pages may count twice.
No native Windows/Linux or constrained 8 GB machine was tested. FP32 remains opt-in.

## Compatibility and validation

- Saved precision selects both SigLIP2 towers; existing collections default to INT8.
- CLI search/index precision mismatches fail without replacing the published collection.
- A repeated FP32 CLI index reuses all 128 images and every stage, preserving its generation.
- Missing/corrupt FP32 weights fail rather than falling back to INT8.
- Existing INT8 rerankers and current training exporters reject FP32 collections.
- 266 tests pass; Black, Flake8, Ruff and mypy pass, including Windows/Linux type targets.

## Reproduction and retained artifacts

Use the production-comparison command in the [guide](../siglip2-precision.md)
from a normal terminal with process-tree sampling access. The primary runner
records whether memory measurements are complete. Source/configuration changes
require a new output directory; historical source snapshots remain with each run.

Local artifacts are under `runs/siglip2-precision-v1/`: frozen `subset/`,
per-policy `index-initial.json`, `evaluation/` (initial sandboxed),
`evaluation-native/` (complete memory), `native-model-resources/`, and
`native-verification.json`. Query SQLite/JSONL records, source snapshots,
model/runtime fingerprints and auxiliary audit scripts are preserved.

## Remaining work

Validate on a fresh independently reviewed collection and native Windows/Linux,
measure an 8 GB machine, and evaluate a resident query worker to reduce startup
cost. The full 2,052-image production FP32 index was not rebuilt in this branch;
the earlier full-corpus precision study remains separate diagnostic evidence.
Precision-aware reranker artifacts and any default change require their own evaluation.
