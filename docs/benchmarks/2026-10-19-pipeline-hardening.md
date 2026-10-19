# Image pipeline hardening — 19 October 2026

The frozen 500-image benchmark improved known-positive hybrid Hit@5 from **55.1% to
92.2%** (260→435 of 472 queries). The family-separated holdout improved from
**65.4% to 96.9%** (104→154 of 159). Both runs processed 497 images and correctly
rejected three above the 24MP policy; neither had an unexpected error or timeout.
The extra OCR work increased mean stage latency by **46.4%**.
This is evidence of substantial hardening, with unresolved OCR and multilingual
retrieval limitations; it is not a certification of production readiness.

## What changed

The existing local architecture remains: Pillow decoding, CPU ONNX SigLIP2 visual
embeddings, Tesseract OCR, multilingual E5 OCR embeddings, SQLite FTS and local
fusion. No cloud inference, new serving stack or trained model was introduced.
Production code remains under `src/realopen_lens`; datasets, fixtures, benchmarks,
judgments and evaluation tools live under `benchmarks/` and `tests/`.

- Decode by supported content type, apply EXIF orientation, composite transparency
  onto white, clear stale metadata, close buffers, bound encoded bytes (64MiB)
  and decoded pixels (24MP). Reject invalid limits, nonfinite vectors and invalid
  dimensions; resolve corpus roots and reject escaping symlinks.
- Replace one OCR invocation with bounded evidence-driven candidates: RGB primary,
  opaque-region alternative, sparse layout, grayscale/rule-removal fallback,
  high-resolution text fallback and optional orientation detection. At most four
  recognition attempts, 30s total budget, native call limits and explicit warnings.
  Canonical OCR is retained; confidence-filtered alternate words aid lexical search
  without inflating canonical OCR accuracy or semantic embeddings.
- Preserve visual cosine scores and add one capped OCR contribution. Exact
  identifiers, quoted phrases, corpus frequency and coverage gate lexical evidence;
  correlated semantic and lexical evidence are not double-counted. A nine-point
  development-only grid selected semantic floor .83/margin .04. Holdout did not
  select model settings.
- Add per-item stage isolation, durable SQLite stage/vector writes, retry history,
  native worker supervision/recovery, bounded sequential model residency, memory
  monitoring, immutable configuration/source snapshots, resumable runs and strict
  comparison checks. Missing embeddings remain failed evaluations.

## Dataset and reproducibility

The retained corpus is `data/benchmark-500-v2/`: **500 distinct SHA256 hashes**,
330 real images, 120 authored fixtures, 50 deliberately transformed derivatives.
Seed: **20261019**. Split: 342 development / 158 holdout images; source/template
families and transformations share splits. This avoids exact variant leakage but
is not an independent blind annotation study.

| Category | Images | Coverage |
|---|---:|---|
| Photos | 150 | COCO people, animals, objects, interiors, landscapes, busy scenes |
| Screenshots | 120 | Rich/Textual terminals, IDE, Jupyter, Streamlit, mobile and conversation fixtures |
| Documents | 100 | Tesseract scans, 16 rendered IRS forms, receipts/invoices/forms/reports |
| Diagrams | 80 | C4, Mermaid, Matplotlib; flow, architecture, charts, technical drawing fixtures |
| Challenging | 50 | Five each: blur, noise, JPEG compression, crop, low/high resolution, 90° rotation, alpha, palette, multipage |

Formats: PNG217, JPEG202, TIFF33, WebP24, BMP24. A manifest stores dimensions,
format, hashes, category, provenance/license, parent, split, expected behavior and
available reference text/phrases. Exact duplicates were removed; perceptual flags
and intentional families are retained in `near-duplicates.json`. Related public
UI figures and synthetic templates are not 500 independent content domains.

Sources are pinned repository blobs from Rich, Textual, JabRef, JupyterLab,
Streamlit, Mermaid, Matplotlib and Tesseract, plus C4 CC-BY figures, US government
IRS forms and COCO captions/photos. Per-image COCO license IDs are checked against
official metadata (CC-BY, CC-BY-SA and public-domain choices); licenses, repository
revisions, font license and references live in `benchmarks/sources/`. The builder
uses at most four download workers. A fixed Google Fonts Noto Sans file and Pillow
BASIC layout make authored fixtures deterministic: all 120 regenerated fixtures
were byte-identical in this environment. Poppler versions may change PDF pixels.

The frozen manifest SHA256 is `f0eabd295eaa05e908b1ce6e47b160d27f433157b0eb1ffef4eba7c7176aae30`; its copy is
`benchmarks/sources/dataset-v2.lock.json`. New builds use v3 annotations, described
below; do not compare a newly built manifest to frozen v2 results as identical.

## Reference audit and evaluation rules

A visual audit found a basename-only annotation bug: Mermaid `header.png` had
inherited an unrelated header example's phrase/query. Its low-resolution child
inherited the phrase too. Both invalid OCR references were removed and the one
invalid query was explicitly excluded. A Jupyter menu reference was corrected
from “Remove Shortcut” to the visible “Delete Keybinding.” Neither inference nor
rankings changed. The natural-language Jupyter query remains a valid synonym.

[`evaluation-corrections-v1.json`](../../benchmarks/sources/evaluation-corrections-v1.json)
records old labels, edits and reasons. Raw 473-query reports remain intact;
`baseline-audited` and `final-audited` apply identical corrections and retain the
excluded query separately. The comparison checks correction hashes. Dataset
annotations now require the exact repository and source path, with a collision
regression test; fresh manifests are version3.

After audit, 175 images have OCR references; one is intentionally rejected,
leaving **174 OCR-scored images** (157 full text, 17 partial phrase references).
The diagnostic pass gate requires all available token recall, token precision and
phrase recall measures to reach 80%. There are **177 verdicts** including the three
expected rejection checks; **323 images have no accuracy verdict**. Successful
execution on those images is not claimed as correct OCR. CER/WER are reported but
not pass gates because diagrams/forms have ambiguous reading order.

Retrieval uses published captions or authored/visually reviewed known positives,
not exhaustive relevance judgments. Scores mean finding a known positive, not
that every other result is irrelevant. Image families can have multiple positives;
recall differs from Hit@5. Queries comprise 312 natural, 120 exact identifier and
40 translated queries. Translations have not had independent native review.

## Baseline and final results

| Category | Queries | Baseline hybrid Hit@5 | Final hybrid Hit@5 |
|---|---:|---:|---:|
| photos | 166 | 1.8% | 81.3% |
| screenshots | 74 | 78.4% | 95.9% |
| documents | 180 | 85.0% | 98.9% |
| diagrams | 52 | 88.5% | 98.1% |

Challenging variants are relevance candidates for their parent's query, excluding
crops whose original subject may have disappeared. There are no invented labels
for unseen/cropped content. Overall hybrid MRR@10: .430→.834; known-positive
Recall@5: .542→.918. Visual-only Hit@5 stays .513; most retrieval improvement is
fusion correcting OCR dominance, not a better visual model.

| Language | Queries | Baseline hybrid Hit@5 | Final hybrid Hit@5 |
|---|---:|---:|---:|
| en | 320 | 42.5% | 93.1% |
| fr | 136 | 88.2% | 96.3% |
| ar | 16 | 25.0% | 37.5% |

Arabic is a clear remaining weakness: 10/16 queries miss the known positive in
Top5. Actual Arabic OCR has only one untranscribed real source, so multilingual
OCR accuracy is **not established**. French results are dominated by authored
fixtures and do not prove broad real-world coverage.

| OCR diagnostic | Baseline | Final |
|---|---:|---:|
| Passes including expected guard rejections | 151/177 (85.3%) | 166/177 (93.8%) |
| Token recall (micro) | 95.18% | 97.14% |
| Token precision (micro) | 95.17% | 96.16% |
| Key-phrase recall | 303/321 (94.39%) | 311/321 (96.88%) |
| Character error rate | 20.20% | 18.95% |
| Word error rate | 30.88% | 29.90% |

Fifteen diagnostic failures become passes; no pass becomes a failure and no
tracked token/phrase metric drops by more than five percentage points. Gains
are concentrated in authored diagrams: **real-source passes only improve 32→33
of 41 scored**, synthetic110→120/120, derived6→10/13. Those denominators are much
more informative than a blanket claim of 94% real-world OCR accuracy.

The final status ledger is 321ok /176degraded /3expected_rejection. Many degraded
records are empty/low-confidence OCR warnings on untranscribed photographs;
11 are actual referenced OCR diagnostic failures. The baseline emitted fewer
warnings, so comparing raw degraded counts as accuracy is invalid.

## Performance and resource cost

| Stage-sum per image | Baseline | Final |
|---|---:|---:|
| mean | 0.592s | 0.867s |
| median | 0.334s | 0.487s |
| p95 | 2.144s | 3.418s |
| p99 | 4.281s | 6.902s |
| max | 6.780s | 12.466s |

These CPU timings exclude model loading, query encoding and ranking; stage logs
and run events retain wall times. Both runs use two inference threads, one worker
and sequential model residency. Per-image sampled RSS max:797.1→818.5MiB.
Coordinator-plus-descendant peak including loading/query evaluation:
**987.2→979.1MiB**, sampled every20ms, zero monitor access errors. Shared pages
can be counted twice and brief spikes can be missed. This is process memory,
not total system RAM or a proof of reliable operation on every 8GB computer.

Host: Apple M3,16GiB, Darwin25.2 arm64, Python3.12.11, Tesseract5.5.2 with
`eng+fra+ara`; exact dependency versions, language-pack hashes, model artifact
hashes, worker limits and host details are stored in every run config. Models:
SigLIP2 INT8 revision `ba1f3b0843f24bc5417d38e19c37b287d719b2f4`, E5 Small INT8
revision `761b726dd34fb83930e26aab4e9ac3899aa1fa78` (full file hashes in registry).
Inference ran under a macOS network-deny sandbox; a TCP TEST-NET probe returned
EPERM. Downloads are explicit dataset/model setup steps outside inference.
Native Windows/Linux and an actual 8GB device were not available; configured CI
unit-test jobs for all three OSes have not been remotely executed here.

## Failure investigation and iteration

The original source snapshot and all successive full runs are preserved. The
loop included the original baseline, a first improved run that exposed grayscale
OCR regression on Rich traceback, an RGB/region-aware correction and another full
run, then a final full repeat after the supported-format guard. Final repeated
OCR texts, per-image quality, query rankings and query metrics are identical to
the preceding improved full run. A resume invocation reused all completed stages
(500 OCR/vision/semantic records and473 records per query stage); native workers
exited in under one second each and historical memory peak was retained.

Significant diagnosed and fixed causes:

- **Transparency:** Rich log's transparent black border became black during RGB
  conversion and OCR was empty. White compositing plus opaque-region candidates
  recovers95.7% token recall and all four phrases. Controlled RGB/gray/crop runs
  isolate the cause rather than attributing it to the model.
- **Overaggressive grayscale:** first improved traceback recall fell to roughly47%
  from97%. RGB primary plus bounded alternatives restores96.8%; regression evidence
  and the failed intermediate full run are retained.
- **Diagram segmentation:** confident top labels hid omitted lower boxes. Sparse
  layout and thin-rule removal recover authored diagram labels; all120 authored
  fixtures pass the final diagnostic. Dense real C4 still has ordering issues.
- **Rotation and resolution:** bounded orientation retry restores rotated terminal
  text; a larger text-rich fallback improves C4Components token recall from92.4%
  to97.5%. It does not solve exact layout/phrase order.
- **Fusion:** frequent form words such as person/head/business/sign falsely
  displaced photo matches. Visual-score preservation, rare-anchor checks and a
  single capped OCR boost address the systemic issue. Settings were selected on
  development queries; remaining holdout failures were not used for tuning.
- **Recovery:** simulated native hangs/crashes and timeout races now leave durable
  per-item errors, kill only the job process group, restart workers and preserve
  retry history. Error vectors and blocked dependents cannot remain silently stale.

[Structured failure analysis](2026-10-19-failure-analysis.json) retains all11 OCR
failures, all37 hybrid Top5 misses, seven ranking regressions and the15 slowest
images, with IDs, metrics, outputs and artifact pointers. Intermediate OCR
candidates, confidence and boxes remain in each raw individual record/SQLite row.
Selected controlled ablations and diagnostic images are under `.cache/hardening/`.

| Remaining OCR case | Evidence and diagnosis |
|---|---|
| `challenging/blur/041.png` | Bar chart labels are visibly blurred; 11/14 tokens match (78.6%), just below the fixed threshold. Both key phrases survive. |
| `challenging/high-resolution/036.png` | Form 1065 rules/checkboxes become extra digits and mixed-script tokens; token recall 93.4%, precision 79.95%. PDF text-layer reading order differs from OCR. |
| `challenging/low-resolution/030.png` | 180x240 form has roughly 5px-high text; extractor returns empty. Original remains partly readable to a human. |
| `diagrams/c4/Components.png` | 97.5% token recall, but Security and Component appear separated in reading order, failing a contiguous phrase. Dense arrows/rules produce extra tokens and high order-sensitive CER. |
| `screenshots/JabRef-jabref/cf26a4e882-12-05-download-jdk-dialog.png` | Visible Eclipse Temurin combobox row is omitted although Download JDK is found. Controlled fallback variants in ocr-ablations.json recover the row; candidate eligibility/selection remains imperfect. |
| `screenshots/Textualize-rich/dc2a461f12-traceback.png` | Canonical OCR inserts a space into divide_all. Token recall 96.8%; confidence-filtered alternate OCR retains divide_all for lexical lookup, but canonical phrase score correctly remains 3/4. |
| `screenshots/jupyterlab-jupyterlab/34ca6fbad0-4.2-delete-shortcut.png` | Original menu says Delete Keybinding; selected OCR omits the three menu items while reading underlying rows. Reference was corrected from the nonexistent Remove Shortcut phrase. |
| `screenshots/jupyterlab-jupyterlab/a4c3754140-4.2-recent-sidebar.png` | README.md is read as README,md; original sidebar verifies dot punctuation. |
| `screenshots/streamlit-docs/949099216a-badge.jpg` | Visible Needs review is merged as Needsreview. |
| `screenshots/streamlit-docs/98df271482-navigation.jpg` | Visible Log out is merged as Logout. |
| `screenshots/streamlit-docs/b82c652f97-checkbox.jpg` | Mouse cursor overlaps the word model; OCR returns TR. Original confirms partial occlusion, not a missing reference label. |

Seven hybrid MRR regressions remain despite no new Top5 failures in those cases:
C4SystemContext EN, Form941, calculator, status badges, slider, title sizes and
time input. Correct results fall from rank1/2 to rank2/3 because bounded evidence
changes ordering among close visual/semantic candidates. Their full competing
rankings are retained; they were not hidden by averages.

For retrieval misses, channel-level evidence distinguishes fusion failures from
representation failures. The Form1099MSC query finds its positive through semantic
OCR but fusion misses Top5, an unresolved gate/identifier mismatch. Arabic C4/test
page/photo examples fail in all individual channels. Visual inspection of known
positives versus top results confirms multilingual mismatch and ambiguous captions;
these observations do not establish a particular internal model mechanism.

## Artifacts and reproduction

- `runs/hardening/baseline-source/`: original production/worker source.
- `runs/hardening/baseline-final/`, `improved-500-v1/`, `improved-final/`,
  `final-500/`: retained full runs with configs/source snapshots, SQLite,
  `individual.jsonl`, `queries.jsonl`, summaries, failures, logs and resources.
- `runs/hardening/baseline-audited/`, `final-audited/`, `comparison-audited/`:
  corrected evaluation records and before/after comparison.
- `runs/hardening/calibration-v3.json`, `repeatability.json`,
  `environment-network-verification.json`, `probes-final/`: supporting evidence.
- [Public aggregate JSON](2026-10-19-hardening-results.json) and
  [failure JSON](2026-10-19-failure-analysis.json) keep readable evidence in the
  repository. Full local images/models/runs remain ignored to avoid binary bloat.

See [large-benchmark guide](../large-benchmark.md) for build/run/resume commands.
Historical raw runs use frozen v2 data; apply the explicit correction file with
`python -m benchmarks.rescore`, then `python -m benchmarks.compare`. Old configs
include their frozen harness fingerprint; do not resume them with edited harness
files. Use the saved harness/engine snapshot to reproduce an old run, or start a
new directory. Exact source snapshots keep previous behavior reviewable.

Validation: **60 tests pass**, Black, Flake8 (project `tox.ini`), Ruff and mypy pass;
**9/9 real input probes pass** (truncated/empty/disguised/oversized/multipage and
unsupported-content cases). The installed wheel contains production modules and
model registry only; benchmarking is repository tooling. These checks complement,
rather than replace, the full CPU/model benchmark.

## Remaining work

1. Collect independently transcribed real screenshots, invoices/receipts and
   multilingual scans; expand Arabic/native-language queries and review captions.
   Evaluate a fresh family-separated holdout after new changes.
2. Evaluate small-text upscaling, GUI-region selection and box-aware spacing/order
   under the existing attempt/time budget. Cursor occlusion and lost pixels must
   remain explicit limitations; do not fill missing words from labels.
3. Investigate Form1099 identifier handling and the seven ranking regressions on
   new development examples; compare visual/lexical/semantic channels separately.
4. Profile fallback scheduling: real-source mean latency rose .657→1.007s and P95
   2.672→3.957s. Maintain the reliable path while reducing unnecessary retries.
5. Run native Windows/Linux, real8GB devices, longer ingestion sessions and hostile
   input/resource-exhaustion testing. The current suite does not establish database
   scale behavior, all decoder security properties, captions or visual reasoning.
