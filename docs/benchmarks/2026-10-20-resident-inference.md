# Resident query inference evaluation

Date: 2026-10-20. Explicit model reuse through production search.

## Implementation

`SearchSession` owns one serialized subprocess with two lazy text encoders:
SigLIP2 text at the saved precision and INT8 E5. Single-shot search stays available;
INT8 indexing and existing hybrid ranking remain the defaults. The CLI adds
`lens search QUERY --interactive` for repeated queries in one owned session.

Collection artifacts and source files are checked on every request. Only cached
model verification decisions and loaded encoders are reused. Filesystem identity/size/
mtime/ctime changes stop the worker and force full model verification. Timeouts,
crashes and protocol failures reap the worker; the failed request is surfaced and
a subsequent request can restart it. Context exit and explicit close release models.

## Profiling baseline

Six pre-change searches on the same saved indexes identified the encoder path as
the primary cost. INT8 requests took 2.84–4.60 seconds, with 1.72–3.44 seconds in
query subprocess calls and 0.31–0.39 seconds in model verification. FP32 requests
took 4.49–4.69 seconds, with 2.63–3.05 seconds in query subprocess calls and
0.91–1.12 seconds in verification. Hybrid ranking took 0.006–0.021 seconds.
These profiled observations locate costs; cProfile overhead and host load prevent
using them as the controlled before/after timing table.

## Paired production benchmark

Both policies search 2,227 unique images across 5,729 paths, including all 2,052
frozen reference images plus 175 unreviewed extras. Seed 20261020 chooses two
families per kind and retains English/French/Arabic translations: 24 queries in
eight families, six queries each for photographs, screenshots, documents and diagrams.
Each precision runs 24 single-shot requests, one resident startup request and two
complete warm passes of 24 queries: 146 total requests. Thread count is fixed at 2.

All 146 requests succeeded. All 98 resident/single-shot comparisons have **identical
full candidate traces and results**, including scores and order. The two warm
passes each reuse one worker PID per precision. Collection generation identities
and file snapshots remain unchanged. Memory sampling reported zero access errors.

## Latency

| Precision | Single-shot median s | Warm median s | Median speedup | Single-shot P95 s | Warm P95 s |
|---|---:|---:|---:|---:|---:|
| int8 | 1.990 | 0.335 | 5.95× | 2.684 | 0.411 |
| fp32 | 3.623 | 0.375 | 9.65× | 4.174 | 0.442 |

The one resident startup observation is 1.548 seconds for INT8 and 3.382 seconds
for FP32. These are individual observations, not cold-start percentiles. All paired
latencies include source checks, tracing and result validation; benchmark wall time
also includes final resource sampling. OS filesystem caches are not flushed.

Warm verification drops to roughly 1.25 milliseconds median for both policies.
The remaining largest warm stage is source validation: approximately 0.235 seconds
INT8 and 0.236 seconds FP32 across the 5,729 saved paths. This stage stays enabled;
a faster query must not return results from removed or modified files.

## Resources

| Precision | Single-shot peak tree MiB | Resident startup peak tree MiB | Warm peak tree MiB |
|---|---:|---:|---:|
| int8 | 1110.4 | 1307.4 | 1272.8 |
| fp32 | 2951.0 | 2493.6 | 1415.0 |

The encoders remain allocated until session close. Sampled RSS can decline as macOS
reclaims, compresses or swaps pages; declining RSS does not mean the models were
unloaded. This run does not establish an allocation limit or long-duration leak bound.
Multiple sessions multiply workers and retained models. Use one owned session per
bounded integration lifetime, rather than one per individual query.

The machine has 16 GiB RAM and ran on battery. Swap in use ranged approximately
10.5–11.2 GiB. Background load was not isolated. A clean
AC-power run is still needed for isolated performance claims or changing defaults.
Native Windows/Linux inference and 8 GiB hardware were not evaluated.

## Validation and artifacts

Automated validation covers real subprocess timeouts/crashes, oversized/invalid
responses, keyboard interruption, worker reuse and cleanup, bounded concurrent
callers, model mutation, precision binding, lazy graph selection, interactive errors,
changed generations/source files and interrupted benchmark resume without false
warm continuation. Native CLI smoke testing executes two FP32 searches in one
interactive process and exits cleanly.

All 298 tests passed. Black, Flake8 (using `tox.ini`) and Ruff passed. Mypy passed
for the native, Windows and Linux targets across 124 source files. Native inference
validation was macOS only.

The [structured evidence](2026-10-20-resident-inference.json) retains the cohort,
paired checks, per-stage distributions, resource summaries and runtime fingerprints.
Local `runs/resident-inference-v1/` retains initial cProfile data, source snapshots,
full per-request SQLite/JSONL traces and resource records, and command logs.
Use the [resident inference guide](../resident-inference.md) to reproduce the study.
The cohort is diagnostic and previously reviewed; exact parity is the purpose here,
not a new estimate of general retrieval quality.

## Remaining work

Keep resident inference explicit and FP32 opt-in. Re-run under controlled AC power,
validate native Windows/Linux behavior and an 8 GiB budget, and measure longer-lived
sessions before setting service-wide memory defaults. Source validation is the next
latency bottleneck; any caching proposal must preserve stale-file detection.
