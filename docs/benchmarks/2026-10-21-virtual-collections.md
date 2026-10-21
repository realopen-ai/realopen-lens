# Virtual collections validation — 2026-10-21

Named manual collections now organize existing indexed images without moving files
or repeating OCR/embedding inference. The gallery, CLI and local API share the
same transactional organization store and search scope rules. See the
[usage guide](../virtual-collections.md).

## Executed checks

| Check | Result |
| --- | --- |
| Python suite | 463 passed, 1 optional OCR test skipped |
| Chromium gallery suite | 24 passed across desktop/mobile configurations |
| Black / Flake8 / Ruff | Passed; Black checked 178 files |
| mypy | 144 source files, native / win32 / linux targets passed |
| Frontend | TypeScript, Vite build and Prettier passed |
| Native FP32 functional benchmark | 10 cases, 0 failures; 2,227 available indexed images |
| Native INT8 functional benchmark | 10 cases, 0 failures; 2,052 available indexed images |
| Concurrent database initialization | 20 trials of 6 first-use writers; all 6 groups retained in every trial |

Native execution used macOS arm64, Python 3.12.11, ONNX Runtime 1.31.0 and two
inference threads. FP32 refers to SigLIP2; E5 remains INT8. Full source hashes,
pipeline configurations, dependency versions and individual case records are
retained in the benchmark outputs. These are functional checks, not new relevance
judgments or evidence of an improvement in retrieval accuracy.

The ten cases per precision exercise 61-member pagination, a singleton outside
the global top 50, all four text retrieval modes, unchanged full-library traces
and results, an empty scope without encoder startup, indexed-image similarity
with a reference outside the group, and external-image similarity. Filtering
before candidate retrieval is verified rather than filtering the global top 50.

Regression tests additionally cover revision conflicts, idempotent mutations,
foreign-source rejection, unavailable/restored members, membership across file
renames, scope changes during inference, feedback provenance, scoped-training
rejection and bypass of managed full-library rerankers.

## Real gallery checks

A separate source-code server operated on an isolated copy of the FP32 index.
Desktop dark and mobile light sessions exercised creation, selecting two images,
adding membership, scoped text/indexed-image/upload searches, removing a member,
renaming, persistence after reload, confirmed deletion and paging through 61
members. Returned search identities stayed within the selected membership.
Screenshots were inspected; the final sessions recorded no JavaScript errors,
external requests or horizontal overflow.

Browser regressions also exercise stale-revision recovery, long collection names,
selection preservation and dialog bounds while the modal is open.

## Failures found and corrected

- Concurrent first-use requests could inspect a partially initialized schema and
  reject a legitimate database. Initialization now acquires the SQLite write
  transaction before schema/version inspection. A busy optional WAL transition
  retains the valid rollback journal. Concurrent creation has a regression test
  and the repeated native checks above.
- Collection dialogs initially used classes that did not match the shared modal
  styles. Visual inspection exposed an inline/mobile layout failure. Dialogs now
  render through a portal with the shared fixed overlay and responsive dialog
  styles; browser assertions check viewport bounds and overlay positioning.

## Reproduction and retained evidence

From a checkout installed with the project dependencies and downloaded models:

```sh
uv run --no-sync python -m benchmarks.experiments.virtual_collections \
  --source-index runs/personal-index-fp32 --collection personal \
  --output runs/virtual-collections-fp32-new

uv run --no-sync python -m benchmarks.experiments.virtual_collections \
  --source-index runs/retrieval-evaluation-index --collection retrieval-v2 \
  --output runs/virtual-collections-int8-new
```

Each output must be new. The runner copies a published index snapshot, retains
per-case JSONL and summary JSON, and exits nonzero on failed checks. Feedback and
model deployment state are not copied. Source image paths must remain available.

Local artifacts are ignored by Git and retained under
`runs/virtual-collections-v1/`: `validated-fp32/`, `validated-int8/`,
`init-concurrency.json` and `gallery-verified/` (screenshots and task records).
Earlier diagnostic runs are preserved separately in the same directory.

## Limits and next work

Collections belong to one physical source; there is no nesting, cross-source
aggregation or automatic near-duplicate grouping. Membership follows exact file
bytes, not visual equivalence. Scoped feedback is stored/exported with provenance,
but the full-library training workflow rejects it pending scope-aware evaluation.
Managed learned models are bypassed for scoped searches; explicit model overrides
are rejected. No additional model or runtime dependency was introduced.

Native Windows/Linux inference and physical 8 GB validation remain pending.
Collect independently reviewed full-library feedback before selecting or promoting
another learned reranker. Automatic near-duplicate organization remains separate
work.
