# Tesseract text gating and downstream retrieval — 19 October 2026

Guarded gating suppresses false OCR on **11/20** new negative photos while retaining
all 36 new positive transcripts. It recovers seven query successes lost by the
original gate, but finishes at **510/564** known-positive Hit@5 versus **511/564**
for whole-image Tesseract, with higher latency and memory use. Whole-image OCR
therefore remains the default; gating stays opt-in.

This study compares the current whole-image Tesseract pipeline with the same
pipeline plus optional PP-OCRv6 text-region gating. Both use the existing CPU
SigLIP2 visual embeddings, E5 OCR passage embeddings, SQLite FTS5 lexical search,
and evidence-gated hybrid ranking. Neither GLM nor RapidOCR recognition runs.
Masking is not part of this comparison.

## Dataset and evaluation protocol

The corpus contains 556 distinct SHA-256 hashes: the existing audited 500-image
corpus, 20 fresh real photographs reviewed for absence of readable text, and
36 fresh authored screenshot controls. There are 564 known-positive queries:
472 historical, 20 photo captions, and 72 screenshot queries (one request-bearing
sentence and one exact request identifier per image).

The photos come from a fixed-seed selection of licensed COCO candidates outside
all earlier local image datasets. Forty-eight candidates were screened; twenty
were selected. Images with visible or ambiguous text were excluded. Full-size
selected photos were visually reviewed before OCR/detection. URLs, per-image
licenses, checksums, review notes, and exclusions are retained in
[`presence-photos-v1.json`](../../benchmarks/sources/presence-photos-v1.json).
A dHash screen found no fresh-photo pair within six bits of another corpus image;
this is a heuristic check, not proof that scenes are independent.

The screenshot controls cross six authored layout families (terminal, editor,
dashboard, dialog, browser, mobile) with six conditions: small, faint, dark-small,
tiny at high resolution, faint-dark, and edge-positioned text. They intentionally
isolate text scale and contrast, with simple UI framing. They are not captures
from real applications. Authored visible strings supply exact ground truth;
three case-sensitive identifiers per case supply 108 occurrence checks. The
font and its license are pinned. Seed: 20261020.

Labels were frozen before inference and were not changed after observing results.
The photo review was performed by one assistant; no independent human annotator
or native-language reviewer participated. “Fresh” means new to this evaluation,
not a blind or independently collected test set. The six screenshot families and
request-bearing queries limit generalization. Photo captions supply incomplete
known-positive relevance judgments; other results can also be relevant.

The historical reference corrections and excluded Mermaid query are carried
forward unchanged. The original three oversized inputs remain expected rejection
checks. Untranscribed photographs retain unknown OCR accuracy; execution is not
counted as evidence of correct recognition.

## Run method

Both runs independently execute all five inference stages: OCR, image embeddings,
OCR passage embeddings, visual query embeddings, and semantic query embeddings.
Ranking evaluates visual, lexical, semantic OCR, and hybrid channels. The runs
are sequential with two inference threads and sequential model residency. No
previous transcripts, embeddings, or timings are reused. The only processing
change is optional gating before Tesseract; ranking parameters stay fixed.

Inference is enclosed in a macOS network-deny sandbox. Every image/stage result
is committed to SQLite; per-image and per-query JSONL retain actual outputs,
references, metrics, rankings, errors, timing, and sampled process-tree RSS.
Resume checks freeze inputs, source, models, packages, host, and settings.
Both runs use `--variant improved`; the historical `baseline` implementation is
not used as the whole-image comparator.

The existing 80% token/phrase diagnostic is retained for comparability. Exact
case-sensitive identifier occurrence is measured separately and does not change
ranking or automatically mark an image as correctly recognized. Negative controls
pass only when successful OCR returns empty text; an execution failure with empty
output is a failure. RSS is sampled process-tree memory, not total system RAM;
stage-sum latency excludes model loading and query/ranking work.

Dataset manifest SHA-256:
`6955866286178ed1e8f273a2ee0b95a6003c4a2f3eb4837af83bfe8306526d7f`.

## Results

| Metric | Whole-image | Original gate | Guarded gate |
|---|---:|---:|---:|
| Overall hybrid Hit@5 | 511/564 (90.60%) | 503/564 (89.18%) | 510/564 (90.43%) |
| Fresh hybrid Hit@5 | 78/92 (84.78%) | 72/92 (78.26%) | 78/92 (84.78%) |
| Historical hybrid Hit@5 | 433/472 (91.74%) | 431/472 (91.31%) | 432/472 (91.53%) |
| Fresh positive diagnostic passes | 31/36 | 27/36 | 31/36 |
| Exact identifier occurrences | 46/108 | 45/108 | 46/108 |
| Fresh negative images with empty OCR | 0/20 | 12/20 | 11/20 |
| Mean stage-sum latency | 0.839s | 0.950s | 0.998s |
| Median | 0.491s | 0.635s | 0.689s |
| P95 | 3.158s | 3.549s | 3.572s |
| P99 | 6.597s | 6.966s | 7.082s |
| Coordinator + child peak RSS | 876.8 MiB | 1308.1 MiB | 1443.6 MiB |

The safeguard recovers seven successful queries lost by the original gate. All
36 fresh positive transcripts match the whole-image baseline exactly; exact
identifier preservation returns to 46/108. Exact-request search recovers from
29/36 to 34/36, matching whole-image search. This does not imply 34 exact
transcriptions: lexical tokenization can retrieve a request whose hyphen was lost.

Compared with whole-image OCR, guarded gating removes false text on 11/20
reviewed negatives but does not improve overall Hit@5. Its mean stage-sum latency
is 19.0% higher and measured coordinator/tree peak RSS is 64.6% higher. These are
observations on this host, not universal performance estimates. Three expected
oversized-input rejections occur in every run; there are no unexpected image
errors, native timeouts, missing query embeddings, or worker restarts.

The guard is deliberately conservative, including reductions from stride
rounding. The 612px garden-bench photo (`fresh/photos/COCO_val2014_000000089909.jpg`)
becomes a 608px detector input. The fallback restores its noisy OCR, reducing
negative-image suppression from 12/20 to 11/20. Nine cases use the fallback in
total. This cost is retained rather than tuning a scale cutoff to the negatives.

The original Jupyter dialogs Top-5 regression disappears after the safeguard
restores the missing dialog text and its lexical corpus statistics. The French
man-and-horse query remains the sole whole-image-to-guarded Top-5 loss. Other
rank and MRR changes, including regressions below Top-5, remain in the structured
diagnostics. The fresh set still has fourteen known-positive Top-5 misses.

All 500 historical whole-image transcripts reproduce the preceding OCR-only
study byte-for-byte. The expanded candidate corpus can change search rankings,
so this study’s historical query scores should not be substituted for the older
500-candidate headline metrics.

### Fresh screenshot conditions

| Condition (six images each) | Whole passes | Original gate passes | Guarded passes | Whole / guarded exact identifiers |
|---|---:|---:|---:|---:|
| small | 3/6 | 3/6 | 3/6 | 0/18 / 0/18 |
| faint | 6/6 | 6/6 | 6/6 | 10/18 / 10/18 |
| dark-small | 6/6 | 6/6 | 6/6 | 12/18 / 12/18 |
| tiny-high-resolution | 4/6 | 0/6 | 4/6 | 1/18 / 1/18 |
| faint-dark | 6/6 | 6/6 | 6/6 | 11/18 / 11/18 |
| edge-text | 6/6 | 6/6 | 6/6 | 12/18 / 12/18 |

These are diagnostic controls, not independent real-application screenshots.
Passing the broad OCR diagnostic does not override a missing exact identifier.

### Environment and verification

Measured on macOS Darwin 25.2 arm64 with 16GiB RAM, Python 3.12.11, and two
inference threads. Dependencies: ONNX Runtime 1.31.0, NumPy 2.5.4, Pillow 12.3.0,
tokenizers 0.22.2, OpenCV headless 5.0.0.93, psutil 7.2.2. Tesseract version and
`eng+fra+ara` language-pack hashes are retained per OCR record; model hashes and
exact run identities are retained in every configuration.

The three complete runs retain 1,668 per-image records, 1,692 per-query records,
and 8,388 durable stage records. Resume verification reuses all 2,796 corrected
stage records with byte-identical stage payloads, vectors, and query results,
zero stage-history replacements, and the historical memory peak preserved.
The dataset rebuild retained its identical manifest and image hashes.

The benchmarked guard revision passed 102 tests, including real worker-supervision, negative
failure scoring, strict comparison identities, and gate/mask pixel preservation
on the downscaling fallback. Black, Flake8, Ruff, and mypy for Windows/macOS/Linux
targets pass. Subsequent experimental CLI selection checks bring the current
suite to 108 passing tests. The rebuilt production-only wheel and installed CLI are checked
under network-denied inference. Native Windows/Linux model inference and a real
8GiB device remain untested.

## Failure investigation and correction

The first paired run found a real recall regression. All six
`fresh/screenshots/*-tiny-high-resolution.png` controls produced nonempty, useful
whole-image OCR but zero detector regions at the 1600px bound. Gating skipped OCR
and returned empty transcripts. Four of the six previously passed the broad OCR
diagnostic; five previously supported successful exact-request queries. Their
source text remains visible in the retained magnified diagnostic crop.

A replay of `terminal-tiny-high-resolution.png` also returned zero regions at a
2400px detector bound. Therefore resizing is not established as the sole cause,
and increasing resolution is not an established detector fix. This exposes a
small-glyph/model limitation. The implemented safeguard is narrower: an empty
result from a detector that reduced the source's longest edge cannot discard
whole-image recognition. The original Tesseract path runs, an explicit fallback
warning is emitted, and source pixels remain available to subsequent retries.
Mask mode also preserves the source on that fallback. Other empty detections at
unreduced scale can still be wrong; text absence is never asserted as proven.

Two representative remaining negative-image failures were inspected against the
originals and detector overlays:

| Image ID | Evidence |
|---|---|
| `fresh/photos/COCO_val2014_000000215245.jpg` | Detector score 0.927 on distant zebra stripes; no readable text in the marked region. Whole-image OCR then invents characters from the scene. |
| `fresh/photos/COCO_val2014_000000225546.jpg` | Detector score 0.812 on an electrical outlet; its holes/patterns trigger the gate. This is not readable printed text. |

A confidence threshold alone cannot distinguish these regions from useful text.
No thresholds were tuned on these examples. Gating is a coarse presence decision:
one false-positive region allows Tesseract to process the whole image.

Recognition itself remains imperfect. Whole-image Tesseract changes `LENS-7300`
to `LENS 7300`, `inventory_sync` to `inventory sync`, and damages slashes in API
paths. These outputs can pass an 80% normalized token diagnostic. Exact identifier
checks deliberately retain these errors instead of treating a diagnostic pass as
an exact transcription.

The initial Jupyter dialogs query regression also exposed a ranking interaction.
Its canonical transcript and semantic similarity (0.864434) stayed identical.
Dropping the tiny authored dialog reduced the lexical document frequency of
`dialog` from six to five in a 556-image index. That crossed the existing rarity
cutoff (5.56) and changed competing authored dialogs' lexical evidence from zero
to 0.8. This is a corpus-frequency discontinuity, not a changed visual encoder or
lost text in the Jupyter screenshot. It is especially visible with six related
fixture templates and should not be interpreted as a representative production
frequency estimate. Ranking weights were left unchanged.

The French man-and-horse photo query loses a semantic boost when gating removes
its nonsensical OCR: target visual similarity remains 0.082919, while the old OCR
semantic similarity was 0.843227. A known-positive ranking can regress when a
spurious OCR advantage disappears. The regression is retained rather than
reclassified as a success.

Original reports remain under `comparison/`. The follow-up full run is under
`guarded-gate/`, with `guarded-comparison/` comparing against the original
whole-image run. The production-source diff is restricted to the OCR safeguard;
encoders, models, fusion parameters, references, and query sets are unchanged.
The harness diff only adds explicit source-change comparison provenance. Both
source identities and a review reason are recorded instead of pretending the
code fingerprints match.

The fresh tiny-text controls became development/regression cases after this
failure was found. Their original `holdout` manifest labels describe the initial
allocation; the final results are not untouched-holdout generalization evidence.
The initial annotation plan's “no tuning” statement does not override this
recorded subsequent use for a code fix. No labels or model thresholds changed.

## Decision and remaining work

Keep whole-image Tesseract as the application default. Gating remains explicitly
opt-in, and masking remains experimental. The new fallback addresses the observed
empty-result regression without changing models, thresholds, or fusion weights.
It does not establish that an empty detector result at unreduced scale is safe.

Next work should use newly human-transcribed real screenshots and independently
reviewed photo negatives, with a fresh evaluation subset not used for fixes.
Prioritize small-text and punctuation/identifier preservation, the rarity-cutoff
and semantic-noise ranking interactions, and reductions in detector overhead.
Do not increase detector thresholds or enable masking based only on confidence.
Validate native Windows/Linux, actual 8GiB hardware, and broader multilingual OCR
before expanding hardware or language claims. All fresh authored text here is English.

## Artifacts and reproduction

- `data/presence-retrieval-v1/`: all 556 inputs, manifest, annotation protocol, and validation checks.
- `runs/presence-retrieval-v1/whole/`: complete whole-image run and source snapshots.
- `runs/presence-retrieval-v1/gate/`: complete gated run and source snapshots.
- `runs/presence-retrieval-v1/comparison/`: per-image/per-query changes and paired reports.

Images, models, and full run artifacts remain ignored. See the
[setup and paired-run commands](../text-regions.md#evaluate-downstream-retrieval).


[Structured results and diagnostic evidence](2026-10-19-text-gating-retrieval.json)
retain the three-way measurements, affected IDs, source review, and validation.

Public ranking examples are limited to the first five results per channel, with
best known-positive positions within the retained Top-50 also recorded. Complete
rankings remain in each run’s per-query JSONL and SQLite records.
