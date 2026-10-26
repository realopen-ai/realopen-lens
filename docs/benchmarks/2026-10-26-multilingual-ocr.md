# Multilingual OCR expansion — 2026-10-26

This initial pack-expansion study predates the optional automatic-mode follow-up.
See the [automatic OCR evaluation](2026-10-26-automatic-ocr.md) for current mode controls
and actual script-detection results. The measurements below remain historical.

## Changes and decision

Desktop builds now ship 19 pinned fast OCR language packs plus orientation data.
The 16 additions total **29.91 MiB uncompressed**. The local macOS development
archive grew from **64.91 to 81.89 MiB** (+16.98 MiB compressed); other platforms
were not built locally. No font assets or new inference dependencies are bundled.

All packs are available offline through the existing language setting. Defaults
remain `eng+fra+ara`; saved selections are preserved. Enabling every pack was
slower and less reliable than explicitly selecting the document languages.
No automatic script/language routing or download manager was added.

Retrieval now preserves meaningful Indic vowel marks/composed Hangul, normalizes
OCR-inserted Han/Kana spacing and adds bounded character terms for unsegmented
phrases. Older saved FTS indexes are rebuilt in memory, without changing their
immutable files. New publications carry a tokenization version. Runtime identity
also covers the new text-normalization module for learned-ranker compatibility.

Korean automatic/sparse layout fragmented text into confident-looking pieces.
A uniform-block probe recovered it. Production now offers one Korean block retry
only when the Korean pack is selected and Hangul is actually observed, within the
existing four-attempt/time budget. The block result can replace fragmented output
when it retains at least 85% of readable-evidence score and gains five confidence
points, or has a higher evidence score. Other language selections keep the prior
attempt behavior. This is a fixture-supported heuristic, not a general accuracy
guarantee.

## Dataset and method

57 synthetic images: 19 language configurations × document, dark terminal and
compressed/slightly blurred/rotated ticket layouts. The corpus covers Latin,
Cyrillic, Arabic, Devanagari, Bengali, Han/Kana and Hangul. Chinese writing variants
are counted as two language-pack configurations. Source strings, font URLs,
licenses, checksums, queries and diagnostic pairs are committed under
`benchmarks/sources/multilingual-ocr-v1/`; generated pixels and manifests remain
under `data/benchmarks/multilingual-ocr-v1/`.

Google Noto fonts are pinned. Existing Chromium tooling renders contextual/RTL
shaping; no shaping dependency is added to the application. These are controlled
assistant-authored transcription references, not independently reviewed translations
or representative real-world language distributions. Chinese variants share a
rendering font; regional font diversity is not evaluated.

Each OCR run has 228 individual records, retained in SQLite and JSONL. Recognition
uses production OCR, two threads, serial isolated workers and fixed interleaved
profile order. Each item has a 50-second supervisor deadline. Engine/pack/source
hashes, environment, output, attempts, errors, latency and sampled process-tree RSS
are retained. Repeating the identical command resumes; changed input/configuration
is rejected. A completed-run resume was exercised without adding result rows.

CER retains diacritics, folds case/compatibility characters and excludes whitespace
and bidi format controls. Scores are uncapped. Execution errors and degraded
transcripts are separate; the diagnostic degradation threshold is CER > 20%.

Profiles:

- **Baseline:** existing eng/fra/ara selection, including on unsupported-script inputs.
- **Matched:** known reference language plus English; an oracle configuration.
- **Script:** known reference script group; also an oracle, not production detection.
- **All:** all 19 packs together.

## OCR comparison

| Profile | Before CER | Final CER | Final median latency | Final P95 | Maximum sampled tree RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 35.51% | 35.51% | 0.488 s | 0.781 s | 215 MiB |
| Matched | 6.20% | 5.00% | 0.343 s | 0.541 s | 212 MiB |
| Script | 8.12% | 6.80% | 0.611 s | 1.342 s | 319 MiB |
| All | 6.76% | 6.54% | 1.083 s | 2.994 s | 482 MiB |

Both valid complete runs processed all 228 items with **zero execution errors**.
Korean CER with its targeted pack selection fell from **50.00% to 4.29%** on its
three regression layouts. All-pack Korean CER remained **41.43%**; merely loading
more languages is not a substitute for a suitable selection.

Timings include isolated Python worker startup, engine initialization and retries.
These are local CPU observations, not a controlled statistical latency claim.
20 ms RSS samples include coordinator/worker/OCR descendants; shared pages can be
counted more than once and brief peaks can be missed. Per-language aggregates are
in the [machine-readable report](2026-10-26-multilingual-ocr.json).

## Retrieval and production workflow

The eleven frozen lexical target/distractor cases improved from **4/11 to 11/11**
with the intended target at rank one. Strict exclusion improved from **4/11 to
9/11**: two Chinese distractors still match part of the query under existing OR
semantics, below the stronger full target. This is retained and documented rather
than scored as an exact-phrase filter.

With the same pre-fix matched OCR, tokenization changes raised lexical MRR@10 and
single-reference candidate Recall@50 from **0.7895 to 0.9474**. With final OCR they
reached **0.9649**. Thus the tokenization and OCR effects are distinguished.

A production-worker run indexed all 57 images as 19 language folders with INT8
SigLIP2 and E5, with zero failed files. All **171 saved-library lexical, semantic
and hybrid requests** completed without errors:

| Mode | Queries | Candidate Recall@50 | MRR@10 | nDCG@10 | Median request |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical, own reference | 57 | 0.9649 | 0.9649 | 0.9649 | 0.021 s |
| Semantic, cross-language topic | 57 | 0.9981 | 0.9576 | 0.9286 | 0.165 s |
| Hybrid, cross-language topic | 57 | 0.9727 | 0.9567 | 0.7967 | 0.214 s |

Cross-language rows exclude the query's own language and use three coarse topic
families as positives. Shared numbers and synthetic layout/background cues make
these easy diagnostics. They are not an independent relevance holdout, do not
establish general visual multilingual accuracy, and are not grounds to promote a
learned model. E5 artifact/tokenizer hashes are recorded; its model is unchanged.

A separate lexical-only replay of the 357 historical queries against the existing
2,227-image corpus preserved aggregate known-positive metrics: Recall@50 0.476190,
MRR@10 0.358485 and nDCG@10 0.372063. 353/357 candidate orders were identical.
BM25 scores changed on 350 queries because indexed terms/document-length statistics
changed; the four changed orders and all per-query scores are retained. This was
not a full before/after hybrid-model inference benchmark.

## Failure analysis and remaining work

Six final matched cases exceed 20% CER:

| IDs | Observed issue | Diagnosis/status |
| --- | --- | --- |
| `ara-0`, `ara-1` | Missing header or instruction line | Retained in image; direct block probes also miss it. Engine/layout limitation remains. |
| `ben-1` | Missing third instruction line | Glyphs are visible and correctly shaped; automatic/sparse/block probes miss it. Unresolved. |
| `fas-0`, `fas-2` | Missing/fragmented invoice or ticket lines | Existing layout modes and font/model interaction; unresolved. |
| `jpn-0` | Incorrect/missing invoice characters | Pack/font/recognition limitation remains; no claim of fully accurate Japanese OCR. |

Images and intermediate outputs for these cases were inspected. The Korean
regression fixture is committed under `tests/fixtures/multilingual-ocr/`.
Native regression, language-change cache reuse, new-pack integrity, legacy FTS,
Unicode equivalence, CJK spacing, query bounds and unchanged default tests were added.

An initial benchmark setup omitted the custom tessdata directory's TSV config;
all 228 calls failed header validation. That invalid run is preserved and excluded
from comparisons. Preparation now supplies the config and the runner fails early
if it is absent. An intermediate Korean selection run is retained but excluded;
final transcript selection was corrected before the complete repeated run.

Broader independently reviewed documents/screenshots, vertical CJK text, handwriting,
regional fonts and mixed-script real images remain unvalidated. No model was trained
or promoted. Native Windows/Linux archives need their platform CI execution.

## Validation and artifacts

- 859 Python tests passed; one optional test skipped. The native pinned-pack Korean
  regression was explicitly enabled and passed.
- 102 desktop/mobile browser tests passed.
- Black, Flake8, Ruff, mypy, TypeScript and existing gallery build checks passed.
- Rebuilt macOS archive passed SHA-256 and twelve frozen-package checks, including
  initialization of every shipped language pack with host OCR hidden. A separate
  frozen-app Korean index/search check also passed with host tools/data hidden
  (3.57% CER, one lexical match).
- Raw baseline/final SQLite, JSONL, per-query rankings, source snapshots, resource
  samples, failure cases and logs are retained under `runs/multilingual-ocr-v1/`.
