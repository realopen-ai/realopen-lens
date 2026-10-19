# Local OCR comparison — 19 October 2026

Tested **RapidOCR 3.10.0 / PP-OCRv6 small** against the existing Tesseract pipeline
on the same frozen 500-image corpus. Both engines ran fresh, sequentially, with two
inference threads on the same Apple M3/16GiB host under a macOS network-deny sandbox.
Production code and its default OCR engine were not changed.

RapidOCR is a promising optional screenshot backend, not yet a drop-in replacement:
real-source diagnostic passes increase **33→38/41**, but memory increases about
sixfold, several document metrics regress, and this selected model lacks Arabic.
The next experiment should cap detector resolution and add an Arabic recognizer
before considering a production switch.

## Quality

Only 174 images have usable OCR references; untranscribed images are not counted
as accurate. The fixed diagnostic gate requires 80% on every available token
recall, token precision and key-phrase recall metric. CER/WER are diagnostics,
not pass gates. All output, including errors and degraded cases, is retained.

| Metric | Tesseract | RapidOCR |
|---|---:|---:|
| OCR diagnostic passes | 163/174 | 170/174 |
| Token recall | 97.14% | 96.72% |
| Token precision | 96.16% | 98.19% |
| Key-phrase recall | 96.88% | 98.75% |
| Character error rate ↓ | 18.95% | 22.26% |
| Word error rate ↓ | 29.90% | 34.12% |

| Source stratum | Tesseract OCR passes | RapidOCR OCR passes |
|---|---:|---:|
| real | 33/41 | 38/41 |
| synthetic | 120/120 | 120/120 |
| derived | 10/13 | 12/13 |

| Category | Tesseract OCR passes | RapidOCR OCR passes |
|---|---:|---:|
| challenging | 10/13 | 12/13 |
| diagrams | 22/23 | 22/23 |
| documents | 98/98 | 98/98 |
| photos | not evaluated | not evaluated |
| screenshots | 33/40 | 38/40 |

All counts above exclude expected decode-policy rejections. Each engine inferred
497 images and correctly rejected three above the 24MP policy; neither had an
unexpected runtime error or timeout. The portable aggregate additionally retains
the three guard verdicts for comparison with the previous hardening report.

## Performance

| Per-image OCR pipeline time | Tesseract | RapidOCR |
|---|---:|---:|
| mean | 0.766s | 0.731s |
| median | 0.407s | 0.416s |
| p95 | 3.257s | 2.521s |
| p99 | 6.807s | 5.680s |
| max | 12.206s | 7.291s |

Timings include decoding and OCR postprocessing, exclude engine startup/warmup,
and include all 500 records (including the three fast expected rejections).
Environment, initialization duration, model hashes, parameters and 20ms sampled
worker process-tree RSS are retained in the accompanying JSON. RSS includes
resident models and OCR children, not just allocation per image; shared pages may
count twice and short spikes may be missed. This was not an actual 8GB-device test.
OpenCV retains its default 8 helper threads; two configured inference threads are
not a hard total-process CPU limit. Timings are indicative of these configured
pipelines, not a strict two-core microbenchmark. The models total about 30MiB on
disk; OpenCV and other optional package dependencies
also add installation size. Production wheel dependency metadata still excludes
RapidOCR/OpenCV and benchmark code.

Peak sampled worker RSS: **462.6MiB (0.45GiB) Tesseract →2832.9MiB (2.77GiB)
RapidOCR**. Both monitors report zero access errors. Small weight files do not imply
small inference memory. RapidOCR's default detector preserves large images and
upsamples small ones (`limit_type=min`); the 2400px input cap is not a hard bound
on internal detector tensors. Bound the detector's longest side and re-evaluate
accuracy before targeting an 8GB machine.

## Diagnostic changes

| Image | Verdict change |
|---|---|
| `challenging/blur/041.png` | fail → pass |
| `challenging/high-resolution/036.png` | fail → pass |
| `screenshots/JabRef-jabref/cf26a4e882-12-05-download-jdk-dialog.png` | fail → pass |
| `screenshots/Textualize-rich/dc2a461f12-traceback.png` | fail → pass |
| `screenshots/jupyterlab-jupyterlab/34ca6fbad0-4.2-delete-shortcut.png` | fail → pass |
| `screenshots/jupyterlab-jupyterlab/a4c3754140-4.2-recent-sidebar.png` | fail → pass |
| `screenshots/streamlit-docs/949099216a-badge.jpg` | fail → pass |
| `screenshots/streamlit-docs/98df271482-navigation.jpg` | fail → pass |
| `screenshots/streamlit-docs/edf6779c3b-radio.jpg` | pass → fail |

Eight failures become passes, and one previous pass fails. The new phrase error
is the Streamlit radio example: visible **Goldfish** becomes **Goldtish**. Fixed
real cases include the JDK vendor row, traceback identifiers, Jupyter menu items,
README punctuation, badge spacing and Log out spacing. Four failures remain:
the tiny form, C4 component phrase ordering, cursor-occluded checkbox and Goldfish.

Do not reduce the result to the higher pass count. **Seven images have a token or
phrase metric drop exceeding five percentage points**, including Form W-4 token
recall 98.2%→86.8%, SS-4 recall 98.6%→91.9%, Form 1099MSC recall 98.3%→90.8%, a rotated
fixture and two authored graphs. See the public JSON for every metric regression.

Overall CER worsens 18.95%→22.26%, WER 29.90%→34.12%, and token recall 97.14%→96.72%.
Inspection of Form 4868 shows RapidOCR interleaving two columns: CER 9.4%→59.8%
while token recall remains 97.4%. That case contributes 2794 of the net 3638 extra
character errors. Reading order explains much of the CER increase, while the
W-4/SS-4 token drops demonstrate real content loss as well. A pass above the 80%
threshold does not erase those regressions.

The comparison is exploratory: these cases have already been inspected during
Tesseract development. No RapidOCR settings were tuned on the holdout, but this
is not a fresh blind test. Both engines receive the same audited references.
The [historical reference corrections](../../benchmarks/sources/evaluation-corrections-v1.json)
remove an unrelated Mermaid header phrase and correct the Jupyter menu action.

Selected difficult-case metrics are retained in the full per-image JSONL; the
machine-readable public aggregate contains every verdict change. RapidOCR outputs
line polygons; Tesseract outputs word boxes. Their raw confidence scores are not
calibrated to each other and are not a substitute for reference-based accuracy.

## Scope and reproducibility

See [OCR comparison commands and limits](../ocr-comparison.md). The optional
`ocr-eval` dependency group pins RapidOCR 3.10.0, whose bundled detector, multilingual
recognizer and line-orientation model are verified against SHA256 values in
`benchmarks/sources/rapidocr.lock.json`. Explicit local model paths and the OS
sandbox prevent inference-time downloads; one multilingual recognition model is
used for every image, without selecting a script from the ground-truth labels.

All 1000 engine/image results are in `runs/ocr-comparison/ppocrv6-small-v2/results.sqlite`
and the two 500-row JSONL files, with OCR output, expected text/phrases, metrics,
latency, RSS, errors and metadata. Per-engine logs, configuration, recovery events,
changes and aggregate reports are alongside them. Source and dependency snapshots
are retained with the completed experiment. Previous Tesseract/500-image hardening
runs remain untouched. [Portable results JSON](2026-10-19-ocr-comparison.json)
retains category/format/resolution/difficulty/source metrics and changes.

This run compares OCR only; downstream retrieval gains were not tested. The tested
PP-OCRv6 recognition routes include English/French and other languages, but **not
Arabic**. On the untranscribed Arabic sample, Tesseract reads `العربي`, while this
RapidOCR configuration returns empty. That is a script-support gap, outside the
scored accuracy denominator. A separate Arabic recognition model (e.g. PP-OCRv5
Arabic) or a Tesseract fallback is required; automatic language routing was not
implemented. Native Windows/Linux and an actual 8GB device remain untested.

The first trial retains 152 Tesseract records at `runs/ocr-comparison/ppocrv6-small/`.
Its heartbeat-gap supervision problem was fixed and regression tested before the
complete fresh run. Repeating the completed command reused all 1000 results without
loading models again or overwriting the measured memory peaks. Source/dependency
snapshots are retained beside the completed run.

Validation: **64 tests pass**, Black, Flake8, Ruff and mypy pass. The production
wheel still excludes the optional engine and its dependencies.

Official model and licensing references:
[RapidOCR model routes](https://rapidai.github.io/RapidOCRDocs/main/en/model_list/),
[local model loading](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/how_to_use_offline_model/),
[upstream attribution](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/).
