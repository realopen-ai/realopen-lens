# Cross-folder virtual collection validation

Library collections span saved source folders while keeping existing folder collections compatible. Membership is content-based, and queries filter candidates before ranking. No encoder, ranking model, dependency, or source image is changed.

## Checks

- 757 Python tests passed; one optional test skipped.
- All 68 desktop/mobile browser tests passed. Coverage exercises library selection, collection creation, adding members, browsing, unavailable notices, scoped text/similarity/upload requests, rename, and removal.
- Regression cases cover membership revisions, pagination snapshots, source rename/refresh/unregistration, empty collections/workspaces, exact-pixel aliases, isolated query feedback, CLI CRUD/search, and transactional removal of more than 500 historical aliases from a bounded logical selection.
- A 65-image regression verifies that a member outside the library top 50 is retrieved when querying its collection.

## Native-model checks

An isolated copy of the existing 2,227-byte-identity mixed INT8/FP32 workspace was used. A collection contained one image from each precision partition. Eight recorded checks passed: browsing, four text modes, indexed-image similarity, queued API/SSE retrieval, and source unregistration. Source rename was checked before removal. Text results never escaped the two saved members; similarity excluded its query and returned the other member. Source removal retained the missing member and reported it as unavailable.

The usable resource-sampled run is retained at `runs/cross-folder-collections-v1/native-02.json`; its isolated index and reproducible smoke script are retained alongside it. All measured cases had zero memory-sampling access errors. The earlier `native.json` run also passed functional assertions, but sandbox process access restrictions made its process-tree memory measurements incomplete; those memory values are not used below.

| Case | Elapsed seconds | Peak process-tree RSS (MiB) |
| --- | ---: | ---: |
| browse | 0.570 | 104.6 |
| text-hybrid | 7.989 | 1815.4 |
| text-visual | 4.558 | 1758.0 |
| text-semantic | 1.835 | 1289.0 |
| text-lexical | 0.690 | 1293.4 |
| similar | 2.782 | 1202.1 |

These are diagnostic functional checks, not relevance benchmarks or isolated performance measurements. Other validation processes ran on the same host. Process-tree RSS may count shared pages more than once. The existing 357-query text evaluation is unchanged and was not rerun for this membership feature.

## Limits

- Mixed-precision visual and image-similarity ranking remains experimental.
- Exact-pixel alias continuity requires published duplicate scans/cache identities. Near-duplicate review does not merge membership.
- Missing identities remain stored; aggregate availability counts may exceed the number of displayed cards when previously distinct identities collapse after a duplicate scan.
- Collection-scoped/library feedback remains exportable but is rejected by the existing full-folder reranker training protocol.
- Workspace import remains limited to supported single-folder sources; cross-folder collection import is refused explicitly.
- Native Windows/Linux execution was not repeated. The macOS arm64 app was rebuilt: its archive checksum, 12 frozen-package checks, real-model indexing/search/refresh/gallery smoke, and frozen library collection creation/browsing passed.
