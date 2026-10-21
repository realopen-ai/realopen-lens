# Feedback-training workflow validation

This validates the local training workflow, not a search-quality improvement. The
native fixtures contain six generated English documents with six explicit
synthetic labels, divided into one train, one validation and one test query. They
are infrastructure checks, not independent human relevance judgments.

## Implemented behavior

- Freeze reviewed feedback exports, group assignments and checksums. Reject stale
  revisions, incompatible inputs, query-family/source-group split leakage and
  edits to frozen inputs.
- Share the existing NumPy logistic optimizer between production and benchmarks.
  Select regularization on validation candidates before production test queries.
- Preserve the hybrid top-50 set; explicitly bind new model artifacts to INT8 or
  FP32 encoder precision. Legacy artifact behavior remains covered by tests.
- Retain individual paired production results, failures, known-judgment metrics,
  validation/test summaries, latencies and resource-measurement limitations.
- Explicitly activate, disable and roll back collection-scoped versions with
  artifact/runtime checks and an audit history. Diagnostic data or query
  regressions require an experimental acknowledgment.
- Expose reviewed preparation/training through CLI and the local gallery, using
  the existing bounded task queue, cancellation and SSE progress.

## Executed checks

| Check | Result |
| --- | --- |
| Python suite | 445 passed; 1 optional OCR experiment skipped |
| Chromium desktop/mobile suite | 16 passed |
| Black / Flake8 / Ruff | Passed |
| mypy native / win32 / linux | 139 source files passed for each target |
| TypeScript / formatting / Vite build | Passed |
| Native INT8 workflow | 6 documents indexed; 2 paired evaluation queries; 0 evaluation failures |
| Native FP32 workflow | 6 documents indexed; 2 paired evaluation queries; 0 evaluation failures |
| Installed-package API/SSE workflow | Reviewed preparation, real training, guarded activation, search model provenance and restore-to-hybrid passed |
| Native gallery inspection | Desktop/mobile screenshots inspected; no horizontal overflow, JavaScript errors or external requests |

The final installed API run delivered seven training SSE snapshots. Short optimizer
iterations may complete between snapshots; inference progress remains observable
through actual search-stage updates. Synthetic activation/rollback checks use an
isolated fixture index.

## Failures found and corrected

Resource sampling initially allowed an OS process-enumeration permission error to
fail otherwise valid searches. Sampling now records incomplete coverage and
continues quality evaluation. A deterministic regression covers this condition.
Parent and process-tree RSS samples are distinguished; phase-boundary samples are
not claimed as true peak or physical memory.

The learning panel initially inherited a nonwrapping checkbox style and overflowed
the mobile viewport. Scoped wrapping corrected it; the browser regression now
checks the complete activation/disable flow for overflow on both viewports.

Frozen input edits are rejected at training time using preserved review/export
checksums. Cancellation and generation-change regressions verify that incomplete
attempts remain inspectable and cannot be activated.

## Reproduction and retained evidence

With the pinned local models already available, use a new output directory:

```sh
uv run --no-sync python -m benchmarks.experiments.feedback_workflow_smoke \
  --output runs/feedback-training-smoke-int8

uv run --no-sync python -m benchmarks.experiments.feedback_workflow_smoke \
  --output runs/feedback-training-smoke-fp32 --precision fp32
```

Raw local evidence is retained under `runs/feedback-training-v1/`. Final native
snapshots are `frozen-int8/` and `frozen-fp32/`; each contains input documents,
review/export snapshots, selected/trial models, per-query records and a summary.
`api-summary.json`, `training-sse.jsonl`, `gallery-native.json` and desktop/mobile
screenshots retain installed-application evidence. Earlier attempts remain for
diagnosis, including the resource-sampling failures.

Timing distributions over two paired queries are diagnostic and do not establish
a speed advantage. Restricted process inspection made memory coverage incomplete
in the sandboxed CLI checks; this is recorded in the reports. Native validation
was on macOS. Native Windows/Linux inference, physical 8 GB validation and an
independently reviewed multilingual relevance holdout remain pending. No neural
model was trained or automatic promotion performed.

See [the workflow guide](../feedback-training.md) for the review format, commands,
activation rules and interpretation of nonexhaustive relevance judgments.
