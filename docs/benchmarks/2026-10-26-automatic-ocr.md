# Optional automatic script-group OCR — 2026-10-26

## Behavior

Manual selection remains the default, with `eng+fra+ara`. Desktop setup, gallery
workspace settings and indexing options now offer named language checkboxes and an
optional automatic script-group mode. The CLI/API use `languages="auto"`; this
fits existing saved configuration without a database migration. Folder refresh and
watching retain the saved mode. Manual choices survive toggling modes in the picker.

Automatic mode runs local Tesseract `osd` on the whole image and up to two bands,
with a total detector budget of three seconds inside the OCR deadline. Confidence
is a script acceptance score, not a probability. A conservative score of at least
3 is required, with a whole-image result or two agreeing bands. Strong conflicting
groups are combined; uncertain, unsupported or unavailable groups use all installed
supported packs. English is retained when available. The selected packs, probes,
policy hashes and fallback are preserved in OCR output and shown in image details.

The script groups cover Latin, Cyrillic, Arabic, Devanagari, Bengali, Han/Japanese
and Korean. Detection does not distinguish French from Spanish or Chinese writing
variants. Blank/noisy images do not establish text absence. Mixed-script minority
text and confident misclassification remain possible.

## Evaluation

The same frozen 57 synthetic images from the [multilingual study](2026-10-26-multilingual-ocr.md)
were processed using actual detection, without supplying their known language or
script to the selector. Individual outputs, routing decisions, failures, timing,
pack/source provenance and memory samples are retained. No GLM or external service
was used. E5/SigLIP2 models were not changed.

| Configuration | Weighted character error | Median time | P95 time | Maximum sampled tree RSS |
| --- | ---: | ---: | ---: | ---: |
| Targeted manual packs (earlier final run) | 5.00% | 0.343 s | 0.541 s | 212 MiB |
| All 19 packs (earlier final run) | 6.54% | 1.083 s | 2.994 s | 482 MiB |
| Automatic, final policy | 6.61% | 1.153 s | 2.684 s | 510 MiB |

All 57 automatic calls completed without execution errors. 18 used smaller script
groups; 39 used the explicit uncertain-script fallback. Every narrowed route
included the fixture's reference language pack. This is coverage, not proof of
correct script classification: one single-script Spanish image received both Latin
and Arabic packs. The full individual records preserve that over-selection.

An initial lower confidence gate narrowed 38 cases but selected wrong groups for
some short Korean/Latin examples, producing 9.51% weighted CER. These results are
retained rather than reported as successful detection. The final gate reduced those
observed omissions at the cost of more fallbacks. This is diagnostic tuning on
seen synthetic cases, not an independently calibrated classifier or a held-out
accuracy claim.

Seven additional controls used repeated dense Latin/Arabic/Korean text, two mixed
Latin/Arabic/Japanese layouts, a blank page and seeded noise. Dense Latin and Arabic
selected their groups; the other five fell back. These repeated-content controls
do not provide independent language accuracy evidence. Manual configuration remains
preferable when the document languages are known. On this short-text corpus,
automatic mode did not improve latency over simply enabling all packs.

Timings include isolated worker startup, detection, recognition and retries. Memory
is process-tree RSS sampled every 20 ms and may double count shared pages or miss
brief peaks. Runs are local observations, not a controlled statistical performance
comparison. Translations, scripts and fonts remain synthetic/unreviewed as documented
in the source study; no model was trained or promoted.

## Reliability and validation

- Script timeouts, unknown/unsupported scripts and absent group packs have explicit
  fallback records; automatic mode requires `osd` and a supported installed pack.
- Group/policy provenance uses canonical JSON values. A regression verifies that
  persistence and an unchanged automatic run reuse all stage caches; changing mode
  reuses metadata/vision and recomputes OCR/text embeddings.
- Setup checks distinguish manual selections from automatic candidates. UI tests
  cover initial defaults, manual choice retention, indexing requests and actual
  routing/fallback display. Native GUI diagnostics exercise both modes and restoring
  a manual Japanese selection.
- Validation: 877 Python tests passed (one optional skip), 110 browser tests passed,
  and the rebuilt macOS archive passed its packaged-app smoke checks. With system
  Tesseract hidden, three native automatic-mode images completed without failures.
  Updating the routing policy reused metadata/vision and recomputed OCR/semantic
  stages; the unchanged repeat reused all twelve stage results.
- Final individual records are under
  `runs/multilingual-ocr-v1/automatic-release/`; earlier routing runs, controls and
  native acceptance evidence remain alongside them.

The source reference for Tesseract's script confidence and sparse-text behavior is
[its script detector](https://github.com/tesseract-ocr/tesseract/blob/5.5.2/src/ccmain/osdetect.cpp).
The project confidence margin and fallback policy are additional conservative
heuristics. Broader real mixed-script images, native-speaker review and native
Windows/Linux execution remain future validation work.
