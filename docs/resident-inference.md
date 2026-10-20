# Resident query inference

Single-shot `lens search` and `search_collection()` retain their existing behavior.
Use an explicit session for repeated queries to reuse loaded query encoders.
No network listener or detached service is started.

## Terminal

```sh
uv sync --locked
uv run --no-sync lens search "pizza" --index-dir runs/personal-index-fp32 --interactive
```

The first query loads the required encoders. Enter subsequent queries at the prompt;
an empty line, EOF, or Ctrl-C exits and closes the owned worker. Results remain JSON
on stdout; prompts and errors go to stderr. A failed query is displayed and the
session accepts another query; the command exits with code 1 if any query failed.
Each new terminal invocation starts a new session. Precision is read from the saved
collection; INT8 remains the indexing default.

## Python

```python
from pathlib import Path
from realopen_lens.search import SearchSession

with SearchSession(index_dir=Path("runs/personal-index-fp32"), threads=2) as session:
    first = session.search("pizza")
    profile = {}
    second = session.search("a terminal screenshot", profile=profile)
    print(profile)
```

`search()` accepts the usual search options, including mode, limit, candidate
depths, traces and compatible experimental rerankers. Collection, model directory
and thread count belong to the session. The first neural request binds precision;
a different precision requires a new session. Lexical-only searches load no models.
Use the context manager or call `close()` explicitly. Closed sessions reject calls.

The lower-level `search_collection(..., query_encoder=encoder)` accepts an explicitly
owned `ResidentQueryEncoder` for integrations that need to share an encoder across
compatible collections. It has the same fixed model directory/threads/precision
boundary and must also be closed.

## Lifecycle and integrity

- One subprocess per session, at most one outstanding encoding request, and at most
  two lazy text models: selected SigLIP2 text precision plus INT8 E5.
- Concurrent callers serialize; a caller waiting beyond the configured timeout
  receives an error without killing another caller's active request.
- The default encoder request deadline is 120 seconds. Native startup/inference,
  IPC writes and response reads are supervised. Timeout, crash, malformed response
  or invalid embedding stops and reaps the worker. The failing request is raised;
  the next request can start a fresh worker without a hidden retry.
- Model files are fully SHA-256 verified initially. File identity, size, mtime and
  ctime are checked before reuse. A changed signature stops the worker and forces
  full verification. This assumes ordinary local filesystem metadata semantics;
  it is not protection against a malicious privileged writer forging metadata.
- Collection generation/artifacts, source validity, encoder compatibility and result
  thumbnails are checked on every search. Index generations and results are not
  cached. New publications and removed/modified source images remain visible.
- Explicit close stops the worker and releases its models. Normal interpreter exit
  also closes registered encoders; parent stdin closure ends an idle worker.

Resident models stay allocated until close. There is no automatic idle eviction or
cross-process sharing. Multiple sessions can consume multiple workers; integrations
should own and reuse a bounded number of sessions rather than create one per user
request. Loading both query encoders together can increase retained RSS relative
to sequential single-shot processes. This does not establish an 8 GB memory budget.

## Profiling and evaluation

`profile={}` on either search API records index verification, source validation,
model verification, visual encoding, semantic encoding, retrieval, result validation
and total seconds. Encoder timings include IPC and startup where applicable. These
are wall-clock stages, not isolated kernel measurements. Failed calls can leave
partial stage timings; the exception remains authoritative.

Run a balanced deterministic diagnostic against matched saved collections:

```sh
uv run --no-sync python -m benchmarks.experiments.resident_inference \
  --int8-index runs/full-precision-local/control/int8-index \
  --fp32-index runs/personal-index-fp32 \
  --corpus benchmarks/sources/retrieval-v2/corpus.json \
  --judgments benchmarks/sources/retrieval-v2/judgments.json \
  --corpus-prefix retrieval-v2 --allow-extra-images \
  --output runs/resident-inference-local
```

The default seed selects two query families per kind, retaining their translations.
Each precision runs single-shot requests, one resident cold request, then two warm
passes. Configure `--families-per-kind`, `--repeats`, `--threads` and `--seed`.

The output retains configuration/source snapshots, resumable SQLite requests, JSONL
results, per-request stage timings, exact trace/result comparisons, errors and sampled
process-tree RSS. Completed single-shot requests resume. An interrupted resident
sequence is archived and rerun from startup, so a cold restart is not mislabeled
as a continuation of a warm worker. Completed errors remain errors; use a new run
for a separately identified retry study.

Close competing workloads and use AC power for cleaner timing comparisons. Host
load, available memory, swap use and power state are retained per request; these
observations cannot prove an otherwise isolated machine. A single observed resident
startup per precision is not a cold-start latency distribution. Native platform and
constrained-memory validation remain separate from static type checks.

The [executed production comparison](benchmarks/2026-10-20-resident-inference.md)
records exact ranking parity, cold/warm timings, retained memory and host limitations.
