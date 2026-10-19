# Terminal, document, and diagram diagnostic: 2026-10-18

This test extends the photograph sample with nine sourced text-heavy images.
It measures actual OCR text recovery as well as image retrieval. It remains a
small diagnostic set, not a representative release benchmark.

## Corpus and references

The mixed collection contains 59 images:

- 50 COCO photographs from the existing seed-42 sample.
- Three published terminal screenshots from [Textualize/Rich](https://github.com/Textualize/rich):
  a Python traceback, application/server logging, and structured JSONRPC output.
- Three document images from [Tesseract's test repository](https://github.com/tesseract-ocr/test):
  two scanned article pages and the standard printed `phototest` image.
- Three Internet Banking System architecture diagrams from
  [Simon Brown's C4 model examples](https://c4model.com/diagrams/system-context):
  context, containers, and components.

Each downloaded image and upstream document transcript is checksum-pinned.
Document transcripts come from the upstream test set. Terminal and diagram
reference text was transcribed after viewing the images. References were not
derived from the OCR output being scored. Decorative terminal borders and
nontext prompt icons are omitted; diagram reference order lists nodes, then
relationship labels, then footers. Attribution is included with the reference
resources in the package.

There are 77 queries: 50 English photograph captions and 27 text-heavy queries,
one per image in English, French, and Arabic. Query translations were authored
by the assistant and not independently reviewed. Each query has one known
positive, rather than exhaustive judgments across the collection.

All nine OCR reference images contain **English text**. This tests English OCR
and multilingual retrieval; it does not establish French or Arabic OCR quality.

## OCR results

Micro-averaged results across three images in each category:

| Category | Character error rate | Word error rate | Word-token recovery | Exact key phrases recovered |
| --- | ---: | ---: | ---: | ---: |
| Terminal | 35.7% | 49.1% | 81.6% | 8/13 |
| Document | 8.2% | 13.6% | 97.1% | 11/11 |
| Diagram | 60.4% | 82.5% | 90.8% | 10/11 |

CER and WER compare an ordered reference string against the recognized string.
They include reading-order errors and punctuation differences. Diagram labels
can be recognized correctly while appearing in a different linear order, which
explains why their word-token recovery is much higher than their ordered scores.
These rates must not be presented as simple character-recognition accuracy.

Word-token recovery counts matched occurrences regardless of order, retaining
underscores within tokens but ignoring punctuation. Exact key phrases retain
punctuation after Unicode/case/whitespace normalization. This additional check
is useful for code identifiers and error strings that a general word score can
hide. Terminal screenshots were the weakest category on these checked phrases.

The category average hides a complete failure: `terminal/log.png` produced an
empty OCR result, losing all four checked identifiers/phrases. The traceback
also lost the exact identifier `divide_all`; the component diagram missed the
exact phrase `Security Component`.

Separate targeted diagnostics tried Tesseract page-segmentation modes 6 and 11
on the terminal images. Mode 11 recovered `divide_all` in the traceback, but
neither mode recovered text from the empty-result screenshot. Color and
grayscale inversion also failed on that image. These diagnostics did not modify
the baseline pipeline or the results below; the failure remains unresolved.

The baseline uses the existing preprocessing unchanged: Tesseract 5, `--psm 3`,
`eng+fra+ara`, and an OCR image bounded to 2400 pixels on each side. That resize
reduces small diagram labels. The terminal examples contain dark backgrounds,
syntax colors, line numbers, and separate source-location columns. Improving
these cases requires an evaluated preprocessing/layout experiment rather than
assuming the photo benchmark already validated OCR.

## Retrieval results

Known-source-image matches at rank one, with the number of queries in parentheses:

| Query category | Visual | Lexical OCR | Semantic OCR | Equal-weight hybrid |
| --- | ---: | ---: | ---: | ---: |
| Photo (50) | 39 | 0 | 1 | 1 |
| Terminal (9) | 2 | 6 | 6 | 5 |
| Document (9) | 6 | 9 | 8 | 9 |
| Diagram (9) | 0 | 7 | 4 | 5 |

OCR retrieval helps on text-heavy images. Equal-weight reciprocal rank fusion
still performs poorly across the mixed collection: text-bearing images receive
extra channel votes even for photographic queries. Photo top-five recovery falls
from 48/50 with visual retrieval to 1/50 with hybrid fusion in this mixed test.
The earlier photo-only run had fewer OCR-bearing distractors, so this is a
stronger demonstration of the same fusion weakness.

These results motivate two separate follow-up experiments: improving terminal
and small-label OCR, and evaluating fusion that accounts for query intent and
signal quality. Keep the current baseline and use a separate evaluation set to
assess improvements.

## Resources and reproduction

The measured pipeline took about 47.0 seconds and sampled a peak of 989.5 MiB
process-tree RSS on the 16 GiB Apple Silicon host. Sampling recorded no process
access errors. Runtime includes repeated query encodings and reference scoring;
it is not interactive search latency. Memory is sampled every 20 ms and may
double-count shared pages or miss short spikes. An actual 8 GB host remains
untested.

Inference completed under macOS network denial after all assets were downloaded.
The benchmark itself does not automatically detect OS network isolation; external
verification evidence is saved alongside the local report.

```sh
uv run --no-sync python -m benchmarks.samples --count 50 --seed 42
uv run --no-sync python -m benchmarks.ocr_samples --include-photos
uv run --no-sync python -m benchmarks run --data data --queries data/queries.mixed.json --ocr-truth data/ocr-truth.mixed.json --languages eng+fra+ara --corpus-kind public-mixed-photos-terminal-scans-diagrams
```

Assets remain in ignored `data/`. The original report, per-image OCR diagnostics,
and supporting outputs are local files under `runs/mixed-text-heavy/`. See the
[benchmark guide](../benchmark.md) for setup, metric definitions, and offline
verification. Three examples per category, related diagram layouts, and one
terminal-rendering library provide limited coverage; more independent sources
and actual non-English text images are still needed.
