# Tesseract with text-region detection — 19 October 2026

All recognition in this experiment uses Tesseract. No GLM, Ollama or alternate
recognizer was evaluated. The ONNX model performs localization only.

## Full comparison

| Metric | Whole-image Tesseract | Detection + gate | Detection + mask |
|---|---:|---:|---:|
| Main OCR diagnostic passes | 163/174 | 163/174 | 165/174 |
| Small/faint/dark control passes | 18/18 | 18/18 | 17/18 |
| Known no-text controls returning empty | 5/18 | 9/18 | 11/18 |
| Reviewed no-text photos returning empty | 0/6 | 1/6 | 2/6 |
| mean image latency | 0.729s | 0.821s | 0.746s |
| median image latency | 0.386s | 0.464s | 0.392s |
| p95 image latency | 3.121s | 3.464s | 3.091s |
| p99 image latency | 6.330s | 6.703s | 6.560s |
| max image latency | 12.086s | 12.360s | 12.158s |
| Peak sampled process-tree RSS | 495.0MiB | 1291.1MiB | 1458.1MiB |
| Unexpected runtime errors | 0 | 0 | 0 |

Each variant processed the same 500-image frozen corpus plus 36 supplementary
controls, sequentially on the Apple M3 Mac with 16GiB RAM. Main-corpus OCR quality
is scored on 174 referenced images; 323 lack an accuracy reference and three are
intentional size-policy rejections. The controls add 18 authored positive cases
and 18 known no-text cases (12 generated images and six reviewed real photos).
These selected diagnostic negatives do not estimate population-wide false-positive
rates. Controls were used during implementation and are not an untouched holdout.

A positive diagnostic pass requires the same fixed 80% gate on all available token
recall, token precision and phrase recall. Known empty references use a literal
empty-output check; no CER is invented for them. Errors remain failures. The
[structured report](2026-10-19-tesseract-text-regions.json) retains all verdict
changes, metric regressions over five percentage points and grouped metrics.

## Implementation and protocol

The existing whole-image Tesseract path remains available unchanged. Optional
localization uses the pinned, checksum-verified PP-OCRv6 small ONNX detector
(9,929,594 bytes), CPU execution, two native threads, a 1600px longest-edge bound
and at most 2000 contour candidates. A saturated detector falls back to the full
image instead of discarding excess regions. Detector errors remain errors.

Gate mode preserves source pixels when any regions are found. With no detections,
it skips recognition and explicitly warns that text absence is not established.
Mask mode preserves pixels in expanded rectangles and their source coordinates;
it replaces the surrounding background, without perspective warping. Model
scores are localization evidence, not calibrated OCR confidence.

Tesseract 5.5.2 uses the same eng+fra+ara packs and two native threads in every
variant. The detector performs no recognition and does not choose language packs.

A development comparison of 20 main cases and all controls exposed fragmented
small-text detections: masking erased letters in the gaps. Nearby same-line
fragments are now merged, with regression coverage. The affected development
pass count recovered from 18/23 to 20/23, versus 21/23 for baseline and gate.
A framed-text control still regresses, so masking remains experimental.
The initial and corrected development runs are retained separately.

The full run uses networking denied at OS level and an additional Python audit
policy. Workers run sequentially, with 60s item supervision and 7200s per-stage
limits. Individual errors do not stop the rest of the run. Inputs, model hashes,
source snapshots, package versions, settings and raw output are preserved. The
fresh baseline exactly matches all 500 prior Tesseract transcripts. RSS covers
the worker and its Tesseract children; 20ms sampling can miss peaks or count shared
pages twice. These are single-pass timings, not controlled multi-trial estimates.
Actual 8GB devices and native Windows/Linux were not tested.

The exact same command was run again after completion. All 1608 payloads and
stable initialization/resource metadata are unchanged, with zero replacements in
stage history; the supervisor clears only ephemeral current-item heartbeats.
The final implementation passes 88 tests, Black, Flake8, Ruff and mypy. Packaging
and the installed CLI are validated separately from the inference timings.

## Failure analysis and disposition

Gate preserved **all 192 referenced transcripts exactly**, not only their pass
verdicts. Its known-negative false-text count fell from 13/18 to 9/18, while mean
latency increased from 0.729s to 0.821s and sampled RSS from 495MiB to 1291MiB.
The detector therefore adds a measurable resource cost; this is not a blanket
speed improvement. It remains opt-in. One of six reviewed real negative photos
became empty under gating; the other five still produced false text.

Mask reduced false-text controls further, to 7/18, and improved three main
screenshots while regressing one. Main-corpus passes rose from 163/174 to 165/174,
but the framed control regression leaves 17/18 positive control passes. Sampled
RSS rose to 1458MiB. The following originals, detector overlays, masked inputs
and raw attempts were inspected; evidence is retained under the run's
`diagnostics/` directory.

| Case | Evidence and root cause |
|---|---|
| JabRef download-JDK dialog; Jupyter delete-shortcut and recent-sidebar screenshots | Masking recovers the tested phrases and changes fail to pass. These are three actual improvements, not a general claim about all application screenshots. |
| `screenshots/streamlit-docs/edf6779c3b-radio.jpg` | Baseline/gate read `Goldfish`; mask reads `Goldtish`. The entire detected word-region pixels are byte-identical, but surrounding controls/cursor pixels change. Both selected PSM3; mean OCR confidence rises from 76.96 to 93.63 despite the wrong word. This is a recognition-context regression, not missing detection of the word. |
| `positive-14` | The dark framed text changes from `REGION-14` to `|REGION-1 4`. All glyph pixels are retained, but masking leaves a partial frame stroke and changes segmentation/context. Both raw and grayscale attempts show the error. |
| `positive-12` | Some small glyph pixels fall outside the expanded mask. `PostgreSQL` becomes `PostareSQL`; the fixed 80% diagnostic gate still passes. This is a remaining coverage failure and evidence against equating a pass with exact identifier accuracy. |
| `challenging/blur/041.png` | Mask drops the already poorly recognized chart labels; token recall falls from 78.57% to 64.29%. Both variants fail. A detector can discard weak evidence rather than recover it. |
| `challenging/high-resolution/036.png` | The dense form remains a failure; mask precision falls by 5.66 percentage points with additional erroneous tokens. Masking is not a universal document repair. |
| Known no-text controls | Gate fixes four baseline false-text cases; mask fixes six. False text remains on some outlined shapes and real photos. Detector presence scores cannot establish that recognized symbols are meaningful printed text. |

All three paths retain the Arabic probe's `العربي` output. This is one qualitative
probe, not evidence of broad multilingual accuracy. Untranscribed main photos
retain unknown accuracy, even if the detector skips their OCR.

The synthetic controls were used to diagnose and fix fragment masking. The
historical main corpus retains its development/holdout labels, but the small
preflight included some historical holdout cases and the corpus was already used
in earlier engineering work. Results here are comparative diagnostics, not an
untouched-holdout generalization claim. Six real negative controls intentionally
repeat main-corpus photos with explicit no-text annotations; there are 536 cases
per variant, not 536 independent unique images.

The choice is to keep whole-image Tesseract as the application default, offer
**gate** when localization is explicitly enabled, and retain mask as experimental.
Do not automatically select mask based on OCR confidence or benchmark category.
Next validation should expand independently annotated negatives and small/faint
text, measure word/identifier preservation and downstream retrieval, and evaluate
region padding/layout context without sacrificing recall on the rest of the set.


## Artifacts and reproduction

- `runs/text-regions/dev-v1/`: initial 56-case comparison before fragment merging.
- `runs/text-regions/dev-v2/`: repeated 56-case comparison after the fix.
- `runs/text-regions/full-v2/`: all 1608 image/variant records in SQLite and three
  JSONL files, logs, resources, configuration and source/dependency snapshots.
- `data/text-region-controls-v1/`: generated/reviewed controls, pinned font and
  machine-readable provenance, references and checksums.

See [setup and benchmark commands](../text-regions.md). Region detection is opt-in;
Tesseract without a detector remains the default. When enabled, gate is the safer
default; masking must be explicitly selected. Remaining false positives and missed
text should be evaluated against a larger independently labeled dataset before
any default change or automatic routing.
