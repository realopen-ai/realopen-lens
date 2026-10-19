# Three-way local OCR benchmark — 19 October 2026

Tesseract remains the default. GLM-OCR is an explicitly selected secondary backend
through the installed local Ollama; RapidOCR remains benchmark-only experimental.
This report compares complete, retained 500-image passes for all three engines.
GLM has the highest diagnostic pass count, but misses more reference tokens and
is about 11 times slower than Tesseract on this Mac. It remains optional.

## Results

| Metric | Tesseract | RapidOCR | GLM-OCR Q8 |
|---|---:|---:|---:|
| OCR diagnostic passes | 163/174 | 170/174 | 172/174 |
| Completed reference outputs | 174 | 174 | 173 |
| Scored real-source passes | 33/41 | 38/41 | 40/41 |
| Token recall | 97.11% | 96.77% | 94.27% |
| Token precision | 96.18% | 98.21% | 99.47% |
| Phrase recall | 96.88% | 98.75% | 99.69% |
| Character error rate ↓ | 19.14% | 22.22% | 14.82% |
| Word error rate ↓ | 30.32% | 34.03% | 24.90% |
| mean image latency | 0.781s | 0.724s | 8.446s |
| median image latency | 0.419s | 0.415s | 2.667s |
| p95 image latency | 3.263s | 2.528s | 35.213s |
| p99 image latency | 6.670s | 5.491s | 44.711s |
| max image latency | 12.129s | 7.417s | 52.264s |
| Sampled worker/service RSS peak | 0.47GiB | 2.57GiB | 3.05GiB |
| Unexpected runtime failures | 0 | 0 | 1 |
| Truncated output records | 0 | 0 | 0 |

Only 174 images have usable OCR references; **323 images have no accuracy verdict**.
Three intentional >24MP policy checks are reported separately, outside the OCR pass
counts above. Overall summary verdict counts include those three guard checks.
The fixed 80% diagnostic gate requires all available token recall, token precision
and phrase recall measures to pass. A diagnostic pass does not imply an error-free
transcription; CER/WER remain order-sensitive diagnostics. Referenced runtime
failures remain failures in the 174-image denominator; no CER is invented for an
API call that returned no output. The token/phrase/CER/WER table uses the same
173 completed reference cases for all engines (156 full transcriptions
plus 17 partial phrase references). This common-case metric view does not erase
runtime failures from the 174-case pass totals. The public JSON also retains the
original per-engine aggregates and their differing completed-reference counts.

| Category | Tesseract OCR passes | RapidOCR OCR passes | GLM-OCR OCR passes |
|---|---:|---:|---:|
| challenging | 10/13 | 12/13 | 12/13 |
| diagrams | 22/23 | 22/23 | 23/23 |
| documents | 98/98 | 98/98 | 97/98 |
| screenshots | 33/40 | 38/40 | 40/40 |

There are 41 real-source OCR references, 120 authored and 13 derived references.
Do not infer 500-image or broad multilingual accuracy from these denominators.
Some real screenshots have only key phrases, not full transcription references.

The frozen `data/benchmark-500-v2` dataset contains 150 photos, 120 screenshots,
100 documents, 80 diagrams and 50 challenging variants: 330 real-source,
120 synthetic and 50 derived images. It has 342 development and 158 holdout images,
with family grouping retained. Formats are PNG 217, JPEG 202, TIFF 33, WebP 24,
BMP 24. Every entry retains its checksum, source, license/provenance, dimensions,
format, references and difficulty tags. The seed is 20261019. See the
[dataset construction guide](../large-benchmark.md) for sources and reproduction.

## Comparison protocol

All engines process the same frozen manifest and audited references. The CPU
controls are fresh and exactly match the prior retained Tesseract/RapidOCR texts.
The merger validates the image manifest, reference edits, shared decoder and
scoring implementation, dependencies, host, languages and thread configuration.
Each component retains its own source snapshots and configuration hash. No
completed results are replaced by fabricated averages or missing-case exclusions.

Tesseract 5.5.2 uses eng+fra+ara and two native threads; RapidOCR 3.10.0 uses its
verified bundled PP-OCRv6 small models on CPU (two ONNX threads, default OpenCV
helper parallelism). GLM-OCR q8_0 uses Ollama 0.32.14 on Apple M3 Metal, two CPU
inference threads, context 8192, output 4096 tokens, temperature 0 and seed 20261019.
All runs used the same Apple M3 Mac with 16GiB physical RAM, macOS 26.2 and
Python 3.12.11. This is a comparison of local consumer pipelines, not a uniform CPU
microbenchmark. Each measurement is one sequential pass, without confidence
intervals or complete control of background system load.

GLM uses the documented `Text Recognition:` task on the whole decoded image.
It is not the upstream SDK's full PP-DocLayout region-analysis pipeline. Model
manifest digest: 6e5935a916fc8d6878a67c60c2b99648f54e10b4a4af4cbbfb2c308953000e5f.
The server log confirms the vision encoder uses MTL0 and all 18 language-model
layers are offloaded to GPU. All engines share the safe decoder, 64MiB encoded/24MP
decoded policy and 2400px input
longest-edge cap, with engine-specific internal preprocessing. GLM raw text,
including Markdown/HTML and formatting-only responses, is preserved as returned.
No expected words or box/confidence scores are invented.

## Gains and regressions

- Tesseract→RapidOCR: {'fail → pass': 8, 'pass → fail': 1}; 7 token/phrase metric regressions above five percentage points.
- Tesseract→GLM-OCR Q8: {'fail → pass': 10, 'pass → fail': 1}; 11 token/phrase metric regressions above five percentage points.
- RapidOCR→GLM-OCR Q8: {'fail → pass': 3, 'pass → fail': 1}; 16 token/phrase metric regressions above five percentage points.

The [portable result JSON](2026-10-19-three-way-ocr.json) retains all verdict
changes, all token/phrase regressions above five percentage points, per-category,
format, difficulty, resolution and source aggregates, and GLM failures/truncations.
These need to be read alongside the pass totals: a model can lose content while
remaining above the fixed threshold. Formatting/reading-order differences can
also increase CER even where most words are recognized.

## Resources, locality and artifacts

CPU RSS covers each worker and children. GLM RSS also includes the separate
Ollama server and its model-runner descendants, with PID deduplication. Sampling
is every 20ms and can miss short spikes or count shared pages twice. Metal unified
memory is not fully represented by RSS. Ollama model size/size_vram are retained
separately; they overlap RSS and must not simply be added to it. An actual 8GB
machine and native Windows/Linux were not tested.

The dedicated Ollama instance runs on 127.0.0.1:11435 with cloud disabled and one
request/model at a time. Both it and its model children inherit an OS sandbox
blocking all external networking. The Python client additionally permits only
that numeric loopback endpoint. A TEST-NET connection probe returned EPERM;
metadata checks verified loopback access and rejection of other destinations.
Model setup was an explicit download, completed before GLM inference. The user's
normal Ollama service is separate from the benchmark instance. Final cleanup
verified that port 11435 no longer listens, the normal service on 11434 is
available with no models loaded, and the downloaded GLM weights remain installed.

All 1500 image/engine records, available raw output, references, metrics, latency, resource
observations, errors/warnings, logs and configuration are retained:

- `runs/ocr-comparison/three-way-controls/`: fresh CPU controls and source snapshots.
- `runs/ocr-comparison/glm-q8/`: full GLM pass, model identity, server metrics,
  source snapshots and network-policy evidence.
- `runs/ocr-comparison/three-way-combined/`: merged SQLite, three per-engine JSONL
  files and every pairwise change file; component provenance remains explicit.
- `runs/ocr-comparison/glm-smoke/`: representative preflight images and blank probe,
  full outputs, timing and memory evidence, excluded from the 500-image metrics.

See [GLM setup and benchmark commands](../glm-ocr.md). GLM is optional and does not
replace the default Tesseract path. OCR-only quality does not establish downstream
retrieval quality. Model generation can infer unsupported content; untranscribed
photos and the single Arabic word are qualitative probes, not proof of general
Arabic accuracy. Remaining failure diagnoses and next steps are recorded below.

## Investigation and next steps

Original images, raw transcripts and references were inspected for the following
failure groups. Structured diagnoses and reviewed image previews are retained in
`runs/ocr-comparison/glm-q8/diagnostics/`.

| Evidence | Finding and diagnosis |
|---|---|
| `diagrams/c4/Components.png` | GLM recognizes all four key phrases and changes the gate from fail to pass, but recall drops from 97.46% to 82.25%. It omits the visible StatementComponent and email-service node/connector content. Normal stop, not budget exhaustion. |
| `documents/pdf-forms/f1099nec-page2.png` | GLM omits the right sidebar/instructions: 82.56% token recall versus Tesseract 99.49% and RapidOCR 100%. The original RGB/red text is present in the input. Other forms show the same coverage weakness. |
| `challenging/low-resolution/030.png` | All engines fail the gate. Tesseract is empty; GLM substitutes an incorrect address, date, identifier and amount. The 180×240 input has lost glyph detail; plausible generated text is not a faithful transcription. |
| `documents/tesseract-ocr-test/600a8d0019-phototest.tif` | GLM adds a fifth quick-brown-dog sentence where four are visible. Token recall is 100%, but precision falls to 86.96%. |
| `screenshots/Textualize-rich/dc2a461f12-traceback.png` | All key phrases recovered, but source line numbers omitted and a Python fence added. The 85.90% token recall reflects structural transcription loss; most actual code is retained. |
| `screenshots/Textualize-rich/dcac557730-logging.png` | GLM repeats/changes timestamps and moves source labels to the end. The output interprets the layout instead of reproducing it verbatim. |
| Six reviewed zebra/surf/pizza photos | GLM returns scene words not printed in the image. Tesseract also emits spurious characters; RapidOCR is mostly empty. These post-run observations are qualitative and do not add references to the fixed score set. |
| `documents/pdf-forms/f941-page1.png` | Initial error reproduced twice: identical 1990-character partial output, `done=false`, no stop reason or generation metrics. The adapter rejects it. The retained benchmark failure is not replaced by retries. The initial raw API JSON was not captured; both diagnostic retries preserve it. |

For Form 941, the repeated dotted leaders and incomplete response are consistent
with the pinned Ollama runner's repeated-token abort path. This is an inference
from the raw response and [Ollama 0.32.14 source](https://github.com/ollama/ollama/blob/v0.32.14/llm/llama_server.go#L1575-L1585);
the internal guard message was not emitted at the recorded log level. It is not
classified as a timeout or a declared output-budget truncation. The exact runtime
branch still needs confirmation upstream.

Main-run inference completion is 497/497 for both CPU engines and 496/497 for GLM,
plus three expected policy rejections per engine. There were no native watchdog
or API transport timeouts. GLM's 320/321 phrase matches and lower common-case CER
come with lower token coverage. Eleven images lose over five percentage points
on a token/phrase metric versus Tesseract; sixteen versus RapidOCR. A higher pass
count is insufficient evidence to promote GLM as the default.

About 81% of GLM's reported completed-inference duration is prompt/image processing
(3380s of 4159s), versus 664s output generation. These server durations exclude
the failed response and are separate from client wall-clock latency. Broad image
indexing with GLM has a substantial throughput cost even with Metal acceleration.

### No-text controls and the narrow fix

Four deterministic 600×600 images, generated without any text, were evaluated
separately after the main run. All original results and the generation script are
retained in `runs/ocr-comparison/no-text-controls/`. They do not change the 500-image
metrics or create corpus-wide false-positive estimates.

| Control | Tesseract | RapidOCR | GLM before guard | GLM after guard |
|---|---|---|---|---|
| White | Empty | Empty | Empty Markdown fence | Empty, inference skipped |
| Gray | Empty | Empty | 9294 characters of repetitive prose; token budget exhausted | Empty, inference skipped |
| Black | Empty | Empty | 561 characters of invented questions | Empty, inference skipped |
| Outlined shapes | Empty | `1` | `O` | `O` |

The new production guard checks exact RGB extrema **before resizing**. If every
pixel has the same color, no glyphs are visible and generation is skipped. It
returns empty text with explicit `inference_skipped=uniform_pixels`, no invented
server stop reason or confidence. It does not use a heuristic near-blank threshold
that might erase faint or tiny text. The three solid controls now return empty
text; the shape false positive remains.

The gray control originally took 41.26s; after the guard it takes about 0.003s
without generation. This is a supplementary single-control observation, not a
new full-corpus performance result. Post-fix outputs are preserved separately in
`runs/ocr-comparison/no-text-controls-after-uniform/`.

All 500 original inputs were rechecked through the shared decoder: 497 decoded,
three expected rejections, **zero exactly uniform images**. Thus the new skip
branch does not apply to this corpus. The full GLM timing/accuracy table remains
the frozen pre-guard measurement; a second full 500-image inference pass after
the guard was not performed. Source snapshots retain that distinction.

### Validation and remaining work

The run-identity PID serialization bug was fixed and covered by a test that fails
before the correction and passes after it. The frozen original run resumed using
a retained compatibility shim that only normalizes the PID tuple to a JSON list:
all 500 payloads are byte-identical and no result replacements entered history.
The supervisor cleared its ephemeral current-item heartbeat as expected. Fresh
runs use the corrected serialization directly.

All 1500 merged records retain their component configuration hashes. The final
code passes 78 tests, Black, Flake8, Ruff and mypy. The package builds and installs
without editable paths; the wheel contains production code only. The default
Tesseract CLI and explicitly selected installed GLM CLI were exercised locally.

Keep Tesseract as the default, RapidOCR experimental and GLM explicitly selected.
Before automatic secondary routing, validate text-presence detection on real
negative images, region/layout OCR on a separate development set, transcript
fidelity for identifiers/numbers and an explicit error/retry policy. Do not fill
omissions using references, trust model-generated scene labels as OCR, or accept
`done=false` as completed output. Broader multilingual evaluation, downstream
retrieval impact, actual 8GB operation and native Windows/Linux validation remain
outstanding. The one Arabic word was recognized by GLM in the smoke probe, but
this is not an Arabic benchmark.
