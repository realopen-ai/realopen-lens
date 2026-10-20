# Numeric identifier retrieval evaluation

Date: 2026-10-20. Targeted hybrid-search correction; no encoder, OCR or index rebuild.

## Change and controls

A typed form/invoice/error number explicitly bound to its type or qualifier can
now use the existing exact-code evidence
strength when the first explicit identifier of that type in OCR agrees. The rule
preserves complete compound values, leading zeros and boundaries. English, French
and Arabic markers are supported; typed query aliases cover ASCII, Arabic-Indic
and Persian digits against existing FTS data. Bare numbers, explicit date/quantity
cues and multiple-number ambiguity do not activate the new rule. Numeric punctuation
alone no longer makes dates/amounts exact codes; quoted literals and valid IP/port
identifiers retain their exact-match path.

The evidence ceiling remains 1.4 and the bounded OCR fusion weight remains 0.16.
No weight or threshold was tuned against query outcomes. Candidate budgets remain
50 per channel; INT8 E5, Tesseract, SigLIP2 precision and collection generations are
fixed within each comparison. Both phases use the same resident production runtime.

Both policies search **2,227 unique images across 5,729 paths**, including all
2,052 frozen reference images plus 175 additional unreviewed images. Each baseline
and final run completes all **357 English/French/Arabic queries with zero query
errors**: 1,428 completed baseline/final query records. Per-query scores and labels
are retained. Additional images receive no invented relevance or provenance.

## Full production quality

| Precision | Phase | Recall@50 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|
| int8 | before | 0.9194 | 0.5540 | 0.6006 |
| int8 | after | 0.9222 | 0.5619 | 0.6081 |
| fp32 | before | 0.9899 | 0.8293 | 0.8580 |
| fp32 | after | 0.9899 | 0.8331 | 0.8610 |

No final query regresses in known-judgment Recall@50, MRR@10 or nDCG@10 in
either precision. Visual and semantic channel rankings remain identical for every
pair. These are previously inspected diagnostic judgments, not a fresh holdout or
an exhaustive accuracy estimate. The intended benefit is specific identifier queries.

| Query | INT8 rank before → after | FP32 rank before → after |
|---|---:|---:|
| documents-08-ar | absent top50 → 1 | 9 → 1 |
| documents-09-fr | 6 → 1 | 1 → 1 |
| documents-11-ar | 15 → 1 | 2 → 1 |

These are Form 4868 (Arabic), Form 8822 (French), and Form 940 (Arabic). Other
known-target quality metrics remain unchanged. Full per-query values are retained
in structured evidence; unknown competing images are not labeled as negatives.

| Precision/language | Before R@50 | After R@50 | Before MRR@10 | After MRR@10 |
|---|---:|---:|---:|---:|
| int8/en | 0.9605 | 0.9605 | 0.6655 | 0.6655 |
| int8/fr | 0.9622 | 0.9622 | 0.5900 | 0.5970 |
| int8/ar | 0.8354 | 0.8438 | 0.4064 | 0.4232 |
| fp32/en | 0.9983 | 0.9983 | 0.8696 | 0.8696 |
| fp32/fr | 0.9966 | 0.9966 | 0.8761 | 0.8761 |
| fp32/ar | 0.9748 | 0.9748 | 0.7421 | 0.7538 |

| Precision/kind | Before nDCG@10 | After nDCG@10 |
|---|---:|---:|
| int8/photos | 0.4538 | 0.4538 |
| int8/screenshots | 0.6528 | 0.6528 |
| int8/documents | 0.6559 | 0.6797 |
| int8/diagrams | 0.6872 | 0.6872 |
| fp32/photos | 0.8837 | 0.8837 |
| fp32/screenshots | 0.8056 | 0.8056 |
| fp32/documents | 0.8881 | 0.8977 |
| fp32/diagrams | 0.8466 | 0.8466 |

## Synthetic and literal diagnostics

Thirty literal cases were frozen before ranking changes, including type/value
mismatches, prefixes, leading zeros, ordinary dates/quantities, mixed codes and
Arabic digits. Their original SHA-256 is retained and forced LF in Git. Fourteen
synthetic document images then ran through real Tesseract indexing and production
search. Their generation and OCR transcripts are identical before and after.

All 30 queries have retained traces. Fifteen explicitly targeted positive queries
improve from 10/15 to 15/15 at rank 1. Five improve: French/Arabic invoice and
error queries move 2→1; the Arabic-digit form query moves 5→1. Existing mixed-code
queries stay at rank 1. The other cases constrain evidence activation; absence of
a new identifier boost is not an invented negative relevance label. These authored
fixtures are diagnostic specifications, not general OCR/search accuracy scores.

The portable renderer records actual hashes, Pillow/FreeType versions and reference
byte agreement. Rendering variants are explicit; existing different images are never
overwritten. Before/after runs here use exactly the same frozen image bytes.

## Failure analysis and iteration

- Static review caught numeric IP:port literals being excluded by an initial
  punctuation guard. The partial 220-query run is retained; endpoint regression
  tests were added before full evaluation.
- The next complete INT8 run improved aggregate quality but regressed seven
  queries: `documents-00-en/fr/ar`, `documents-02-en`, and `documents-03-en/fr/ar`.
  Compound `1099-MISC`/`1099-NEC` queries had been reduced to bare `1099`, and later
  form references could boost a different primary form.
- Original Form 8962 visibly identifies itself and says “Attach to Form 1040”;
  Form 4506-T requests transcripts and mentions other form numbers. Their OCR
  preserves those distinctions. The corrected parser keeps complete compound
  identifiers and requires the first explicit identifier to agree. Added regression
  tests cover those exact failure mechanisms.
- A final safety review tightened query-side binding: `invoice for 2026`, its
  French/Arabic equivalents, and quantities mentioned after an error do not become
  explicit identifiers. Five additional negative tests cover these cases. Both
  complete production runs were repeated after this guard; measured gains remain.
- A cached-score replay of all 714 baseline records predicted the correction.
  Both complete native reruns then matched that prediction query-by-query; replay
  values are not substituted for production results. Intermediate full/partial
  runs and diagnosis remain preserved rather than discarded.

A baseline coordinator resumed after SQLite read/write contention, and an
intermediate coordinator resumed after a session interruption. Completed records
were retained under unchanged identities. These are coordinator interruptions,
not hidden query successes or fabricated query failures.

## Latency and resources

| Precision | Phase | Median s | P95 s | P99 s | Peak sampled tree MiB |
|---|---|---:|---:|---:|---:|---:|
| int8 | before | 0.382 | 2.509 | 4.051 | 1084.3 |
| int8 | after | 0.332 | 0.405 | 3.905 | 923.2 |
| fp32 | before | 0.376 | 0.418 | 0.610 | 1563.0 |
| fp32 | after | 0.363 | 0.472 | 0.740 | 1686.3 |

These sequential resident timings include startup/restarts, source/artifact checks,
inference, tracing and memory-sampling completion under uncontrolled host load.
They are descriptive; differences do not isolate identifier-rule overhead. Sampling
has zero access errors but may miss spikes and double count shared pages. Native
inference was tested on a 16 GiB macOS machine; 8 GiB and native Windows/Linux
inference remain unvalidated.

## Reproduction, validation and limitations

Use the [identifier guide](../numeric-identifiers.md) for fixture generation and
production search, and the collection evaluator with `--resident` for both phases.
The [structured report](2026-10-20-numeric-identifiers.json) retains all 714 paired
query metrics and positive ranks, category/language summaries, fixture outcomes,
code/runtime hashes and iteration diagnosis without publishing personal extra paths.
Local `runs/numeric-identifiers-v1/` retains individual SQLite/JSONL results,
candidate exports, reference/source snapshots, exact evaluation scripts and logs.
Ranking changes require a new run directory; completed errors are not silently skipped.

Validation: 369 tests passed. Black (155 files), Flake8 using `tox.ini`, Ruff,
and mypy for native/Windows/Linux targets (126 source files) passed. Coverage
includes full identifiers, first-field references, indirect date/quantity wording,
digit scripts, IP/port compatibility, portable rendering and resident cleanup.

The first-explicit-identifier rule is a heuristic, not page-layout understanding.
Missing/corrupt headings and unsupported marker languages can fail. Multiple-number
queries deliberately fall back. Body references can still qualify if no earlier
identifier is recognized. The broader Arabic-to-English semantic candidate gap
and independent review/new holdout remain separate work. Existing experimental
logistic rerankers were not retrained/evaluated on the changed evidence distribution.
