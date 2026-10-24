# Incremental stage cache validation

Indexing now reuses dependency-scoped stage analyses while preserving existing catalog/manifests and explicit failure handling. No retrieval model or ranking policy changed.

## Automated validation

- 771 Python tests passed; one optional test skipped.
- 74 desktop/mobile browser tests passed, including explicit cached versus processed indexing progress.
- Black, Ruff, Flake8, mypy native/Windows/Linux checks and frontend build/formatting passed.
- Regressions cover new-image-only processing, OCR language/engine changes, separate precision collections, individual model graph changes, semantic text dependency, legacy rows, failed OCR retries, progress accounting, corrupt artifact recovery and avoiding redundant cache writes.

## Native execution

Two generated text-image fixtures were first indexed with the previous frozen macOS app. The new frozen app refreshed that isolated legacy index. A third unique image was then added; language changes, restoration of a previous profile, source/frozen transitions, and a separate rebuild were executed using actual local Tesseract/SigLIP2/E5 models. The user workspace and running indexing job were not modified. These are cache/packaging regressions, not retrieval-quality benchmarks.

| Case | Images | Reused metadata / OCR / vision / semantic | Processed metadata / OCR / vision / semantic | Seconds (whole CLI) |
| --- | ---: | --- | --- | ---: |
| legacy-frozen-transition | 2 | 2 / 0 / 2 / 0 | 0 / 2 / 0 / 2 | 3.474 |
| frozen-unchanged | 2 | 2 / 2 / 2 / 2 | 0 / 0 / 0 / 0 | 0.570 |
| add-one-image | 3 | 2 / 2 / 2 / 2 | 1 / 1 / 1 / 1 | 2.267 |
| change-ocr-language | 3 | 3 / 0 / 3 / 0 | 0 / 3 / 0 / 3 | 1.639 |
| return-to-eng | 3 | 3 / 3 / 3 / 3 | 0 / 0 / 0 / 0 | 0.586 |
| frozen-to-source | 3 | 3 / 0 / 3 / 0 | 0 / 3 / 0 / 3 | 1.916 |
| source-unchanged | 3 | 3 / 3 / 3 / 3 | 0 / 0 / 0 / 0 | 0.549 |
| rebuilt-frozen-unchanged | 3 | 3 / 3 / 3 / 3 | 0 / 0 / 0 / 0 | 2.354 |

The legacy transition conservatively recomputed OCR and semantic data because the older record lacked the new engine/orientation provenance. Both metadata and visual vectors were reused. Adding one image performed one attempt per stage and reused both existing images. Changing languages performed OCR/semantic attempts on three images and zero metadata/vision attempts. Returning to a previously cached configuration reused all stages.

System and private OCR runtimes retain distinct provenance. Their transition reused metadata/vision but recomputed OCR/semantic results; a matching version string is not proof of an identical engine. Subsequent requests with that runtime reused all stages. Source and frozen hybrid search returned identical image ranks from the resulting shared index.

A separate unchanged rebuild also reused every stage: signing/relocation no longer changes the logical private-engine identity. The original source hashes of native binaries still identify genuine engine/library changes, while installed artifact hashes remain available for integrity verification.

## Before/after evidence

The prior full-pipeline implementation was also run with the three-image corpus and a previously unused language configuration: it reused zero stages. In the new implementation, the first language-change case above preserved all three metadata and vision analyses. This eliminated three metadata and three visual-inference attempts while retaining the necessary OCR and semantic work. The separate warmed-profile comparison in `native-before-after.json` demonstrates complete reuse for an already cached new profile; it is not a cold latency comparison.

Whole-CLI timings include verification, process/model startup, scanning and publication. They were collected on a shared development host and do not establish isolated performance or low-memory hardware guarantees. No new relevance metrics were computed.

## Artifacts and limits

Local records and reproducible scripts are retained in `runs/stage-cache-v1/`: legacy baseline, `native-first.json`, `native-rebuild-02.json`, final rebuild records, `native-before-after.json`, `query-compatibility.json`, per-case logs, test logs, and the frozen archive smoke report. Generated fixture images and the isolated catalog remain available there.

- Unknown legacy dependency provenance is recomputed rather than assumed compatible.
- Shared processing/preprocessing implementation changes still invalidate conservatively across affected stages.
- Original generation artifacts and reusable analysis aliases are retained; this change does not add cache garbage collection.
- Runtime tests were performed on macOS arm64. Windows/Linux were covered by type checks and runtime-manifest fixtures, not newly executed native builds.
- The running desktop app was left untouched. Updated desktop artifacts were built and checked separately.
