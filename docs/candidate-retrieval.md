# Candidate retrieval diagnostics

Hybrid search and int8 inference remain the defaults. Candidate depths are now
configurable for experiments; changing a depth does not change an image's
embedding or guarantee a better final ranking. See the
[measured study](benchmarks/2026-10-20-candidate-retrieval.md) before choosing a policy.

## Search controls

```sh
uv run --no-sync lens search "pizza" --index-dir runs/personal-index \
  --visual-candidates 200 --semantic-candidates 100 --lexical-candidates 50
```

Each channel accepts 1–1,000 candidates, independently of `--limit` (1–100
displayed results). Explicit budgets can produce fewer results than the display
limit. Without flags, the existing policy is preserved: 50 per channel, with
the visual depth increasing for display limits above 50. Duplicate image IDs
receive one place in the union. Search still computes exact vector scores over
the collection; larger depths do not require additional query encodings.

The Python entry point exposes the same controls:

```python
from pathlib import Path
from realopen_lens.search import CandidateDepths, search_collection

results = search_collection(
    "pizza",
    index_dir=Path("runs/personal-index"),
    candidate_depths=CandidateDepths(visual=200, semantic=100, lexical=50),
)
```

These controls require hybrid mode. Existing experimental rerankers require
50 per channel and reject other budgets; changing the pool needs a new training
and compatibility study. Both existing training commands also reject wider-pool
exports rather than writing misleading budget-50 models. Traces retain each configured ranking and the
full union; hybrid Recall@50 always evaluates the final first 50, separately
from full-pool recall.

## Reproduce the depth study

Restore and index the [2,052-image corpus](../benchmarks/sources/retrieval-v2/README.md),
then run from the checkout:

```sh
uv run --no-sync python -m benchmarks.experiments.candidate_depths \
  --index-dir runs/retrieval-evaluation-index --collection retrieval-v2 \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --depths 50 100 200 --output runs/candidate-retrieval-v1/depths
```

Each depth performs complete production search requests. SQLite preserves every
query and attempt; JSONL exports include candidates, scores, labels, metrics,
errors, latency and sampled process-tree memory. Repeating the identical command
resumes completed requests. Changes to source, models, generation, judgments,
threads or budgets require a new output directory. `--retry-errors` retries only
failed queries. Comparison rejects incomplete, unscored or mismatched runs.

Commands assume a fresh run directory. Existing historical runs must be resumed
with their exact frozen source or kept separate from a new run prefix.

All queries in this study were previously inspected. The original training,
validation and test roles remain recorded, but none is a new holdout for these
experiments. Defaults must not be selected using these results alone.

## Encoder parity and precision

Model downloads are explicit setup steps. Inference uses verified local files.

```sh
uv run --no-sync lens models fetch --precision fp32
uv run --no-sync python -m benchmarks.datasets.encoder_references
uv venv --python 3.12 .cache/encoder-reference-env
uv pip install --python .cache/encoder-reference-env/bin/python \
  -r benchmarks/sources/encoder-reference.requirements.txt
uv pip install --python .cache/encoder-reference-env/bin/python --no-deps -e .

uv run --no-sync python -m benchmarks.datasets.encoder_probes \
  --index-dir runs/retrieval-evaluation-index --collection retrieval-v2 \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --output runs/candidate-retrieval-v1/parity-inputs
```

Reference models use the pinned revisions/checksums in
`benchmarks/sources/encoder-reference.lock.json`. The isolated environment keeps
PyTorch and Transformers out of application dependencies. On Windows, use the
environment's `Scripts/python.exe` instead of `bin/python`.
The editable, no-dependency application install exposes the local model/offline
utilities to this evaluation environment; it is not the application runtime setup.

Run a separate process for each backend (`int8`, `fp32`, `reference`) and stage
(`vision`, `siglip2`, `e5-query`, `e5-passage`):

```sh
uv run --no-sync python -m benchmarks.experiments.encoder_parity probe \
  --inputs runs/candidate-retrieval-v1/parity-inputs/inputs.json \
  --backend fp32 --stage siglip2 \
  --output runs/candidate-retrieval-v1/parity/fp32/siglip2

.cache/encoder-reference-env/bin/python -m benchmarks.experiments.encoder_parity probe \
  --inputs runs/candidate-retrieval-v1/parity-inputs/inputs.json \
  --backend reference --stage siglip2 \
  --output runs/candidate-retrieval-v1/parity/reference/siglip2

uv run --no-sync python -m benchmarks.experiments.encoder_parity compare \
  --left runs/candidate-retrieval-v1/parity/fp32/siglip2 \
  --right runs/candidate-retrieval-v1/parity/reference/siglip2 \
  --output runs/candidate-retrieval-v1/parity/fp32-vs-reference/siglip2
```

Record native-tokenizer differences first. A separate reference SigLIP text probe
with `--siglip-lowercase` matches production's lowercase inputs and isolates graph
agreement. This flag does not change production preprocessing. Completed probes
can resume when identity and output checksums match. Failed/different probes must
use a new output directory; their per-input records remain available.

Interrupted probes save each completed vector and metadata atomically, then skip
those inputs on resume. The isolated worker has a configurable `--stage-timeout`
(3,600 seconds by default); interruption stops its process tree. Memory records
describe the last worker invocation, while per-input timings can span resumptions.

Embedding agreement alone does not establish retrieval accuracy. To compare
complete precision policies, prepare a second input file with `--full-corpus`.
Run FP32 `vision` and `e5-passage` probes against it, writing to
`runs/candidate-retrieval-v1/precision-probes/fp32/{STAGE}`. The full input file
uses original images and the production character/passage limits. Then:

```sh
uv run --no-sync python -m benchmarks.experiments.precision_retrieval \
  --index-dir runs/retrieval-evaluation-index --collection retrieval-v2 \
  --inputs runs/candidate-retrieval-v1/precision-inputs/inputs.json \
  --query-inputs runs/candidate-retrieval-v1/parity-inputs/inputs.json \
  --fp32-dir runs/candidate-retrieval-v1/precision-probes/fp32 \
  --query-probe-dir runs/candidate-retrieval-v1/parity \
  --baseline-dir runs/candidate-retrieval-v1/depths/depth-50 \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --output runs/candidate-retrieval-v1/precision-retrieval-v3
```

The command verifies input identities/checksums, requires exact int8 production
ranking replay, and retains each policy's complete candidate traces. It reuses
cached OCR and fixed fusion thresholds. Reported ranking times exclude model
loading/encoding; probe timings and memory are separate. FP32 is an evaluation
option here. Production indexing now also supports opt-in
[SigLIP2 precision selection](siglip2-precision.md) with a separate collection index.

## Reviewed expansion and Arabic keyword probes

```sh
uv run --no-sync python -m benchmarks.experiments.query_expansion \
  --baseline-dir runs/candidate-retrieval-v1/depths/depth-50 \
  --output runs/candidate-retrieval-v1/expansion-v2
uv run --no-sync python -m benchmarks.experiments.arabic_lexical \
  --index-dir runs/retrieval-evaluation-index --collection retrieval-v2 \
  --baseline-dir runs/candidate-retrieval-v1/depths/depth-50 \
  --output runs/candidate-retrieval-v1/arabic-lexical
```

Expansion adds at most one reviewed English translation of the same query family.
It measures a translation oracle, not an automatic translator. Max-score fusion
is experimental; the RRF alternative includes its own original-query control so
fusion changes are not mistaken for translation gains. Additional encodings and
larger pools have a cost; reused request times are not measured expansion latency.

Arabic folding adds an experimental keyword view for tatweel, character variants
and digits. Original identifiers remain available. It does not change the saved
production tokenizer or claim stemming/cross-language translation.

## OCR timeout regression probes

```sh
uv run --no-sync python -m benchmarks.datasets.ocr_probes \
  --dataset data/benchmark-500-v2 \
  --extra-image data/retrieval-v2/documents/commons/laurore-zola-1898/88b2b135d7b4690c.jpg \
  --output runs/candidate-retrieval-v1/ocr-inputs.json
uv run --no-sync python -m benchmarks.experiments.ocr_recovery run \
  --inputs runs/candidate-retrieval-v1/ocr-inputs.json --engine-root src \
  --output runs/candidate-retrieval-v1/ocr/band-recovery
```

Use a saved pre-change engine checkout and a separate output directory for the
baseline. The published study retains its engine snapshots. The runner isolates
each image, bounds native execution, retains errors and resumes SQLite records.
Images without valid references receive no invented OCR accuracy score.

Resume identity includes the native executable hash, Tesseract version, language
packs and runtime packages. A changed native environment requires a new run.
Worker pipes use ASCII-safe JSON, while artifacts remain UTF-8, so Arabic output
also survives a Windows-style CP1252 default encoding.

Only after the initial whole-image timeout does production OCR try three disjoint
horizontal bands within the remaining 30-second budget and four native attempts.
Failed attempts count against that limit; PNG encoding counts against each native
deadline. Successful whole-image processing keeps its existing path. Band text
and word coordinates are combined in full OCR-input coordinates, with explicit
partial-output metadata. Unrecovered body text remains missing.

Partial text can supply keyword and semantic evidence but produces a structured
`PartialOCR` issue, a degraded search result and `completed_with_errors` indexing.
It is a completed, reusable partial analysis: ordinary reindexing reuses it, rather
than silently claiming a new full-page recognition attempt. Improving its coverage
requires a new processing configuration/version or a fresh evaluation index.

The compact public evidence can be regenerated from the retained original run tree
(including historical OCR trials and the production smoke-verification record) with:

```sh
uv run --no-sync python -m benchmarks.reporting.candidate_study \
  --run-dir runs/candidate-retrieval-v1 \
  --output docs/benchmarks/2026-10-20-candidate-retrieval.json
```
