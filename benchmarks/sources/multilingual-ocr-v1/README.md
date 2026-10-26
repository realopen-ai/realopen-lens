# Multilingual OCR diagnostic sources

`texts.json`: 19 language codes × three topics (invoice, connection failure, train).
Assistant-authored text and translations are unreviewed. The exact rendered strings
are valid transcription references for these synthetic images; they do not certify
idiomatic translations or general retrieval relevance.

`fonts.json`: pinned Google Fonts files with SHA-256, size and individual OFL URLs.
Font assets remain in the explicitly selected cache. `FONT-LICENSE.txt` reproduces
the OFL text; individual copyright notices remain at each locked source URL.
Rendered images are generated, not font redistributions.

`lexical.json`: diagnostic target/distractor pairs frozen before retrieval changes.
Strict exclusion and target-at-rank-one are reported separately. Two multi-term
Chinese cases intentionally retain partial matches under the existing OR semantics.

Generate using existing frontend Playwright/Chromium tooling (for correct Indic/RTL
shaping), with explicit acquisition:

```sh
npm --prefix frontend ci
cd frontend && npx playwright install chromium && cd ..
PYTHONPATH=src:. uv run --no-sync python -m benchmarks.datasets.multilingual \
  --output data/benchmarks/multilingual-ocr-v1 \
  --assets .cache/multilingual-ocr --fetch
```

The manifest identifies each generated image by SHA-256 and retains source text,
font provenance, category and topic. PNG, dark screenshot and JPEG/blur/rotation
layouts are diagnostic; font rendering is not assumed bit-identical across OS or
Chromium versions. `renderer.json` records the renderer actually used.

```sh
PYTHONPATH=src:. uv run --no-sync python -m benchmarks.experiments.multilingual_ocr \
  --dataset data/benchmarks/multilingual-ocr-v1 \
  --tessdata .cache/multilingual-ocr/tessdata \
  --output runs/multilingual-ocr-v1/ocr

PYTHONPATH=src:. uv run --no-sync python -m benchmarks.experiments.multilingual_retrieval \
  --dataset data/benchmarks/multilingual-ocr-v1 \
  --ocr-run runs/multilingual-ocr-v1/ocr --models models \
  --output runs/multilingual-ocr-v1/retrieval.json
```

OCR executes serially, with two threads per engine, process-tree RSS sampling,
50-second item supervision and per-item SQLite commits. Repeating the identical
command resumes; changed configuration/source/input is rejected. Errors are
retained as failed records. A new output directory is required for changed inputs
or implementation. CER retains diacritics and excludes whitespace/bidi format controls.
Record errors and per-image results alongside aggregates; a successful execution
is not evidence of a correct transcript. CER is not capped at 1.

Profiles: `baseline` selects eng/fra/ara; `matched` selects the known reference
language plus English; `script` uses known script groups; `all` enables 19 languages.
Matched/script are oracle comparisons, not language detection. E5 is INT8 in this
comparison; model and tokenizer hashes are retained. Semantic relevance uses three
coarse synthetic topics and excludes the query's own language; this is diagnostic
cross-language retrieval, not an independent holdout or hybrid visual evaluation.

## Actual automatic routing

Run the same OCR command with `--profiles auto` and a new output directory to
measure production script detection. This mode does not receive fixture language
or script labels. Its detector policy and candidate-pack hashes are bound to the
run, and per-image `selected_languages`/`script_detection` record the actual route.
The reference-based `script` profile remains an oracle. See the separate automatic
OCR report for fallback counts and known limitations.
