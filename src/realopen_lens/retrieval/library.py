"""Global channel ranking over immutable folder snapshots; no per-folder quotas."""

from collections.abc import Callable
from dataclasses import asdict, replace
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from threading import RLock
from typing import Any

from realopen_lens.models.manager import digest, registry
from realopen_lens.organization import duplicates
from realopen_lens.retrieval import memory, ranking
from realopen_lens.retrieval.candidates import CandidateDepths
from realopen_lens.retrieval.search import SearchResult, source_paths, search_results
from realopen_lens.runtime.resident import ResidentQueryEncoder
from realopen_lens.storage.artifacts import load_manifest, matrices
from realopen_lens.storage.catalog import Catalog

LIBRARY_ID = "library-v1"
MAX_IMAGES = 100_000
MAX_FOLDERS = 256


class LibrarySession:
    """One resident worker, one precision at a time; vectors reused within a query.

    Mixed libraries switch workers rather than retaining both model sets in RAM.
    The next query starts with the last resident precision to avoid one reload.
    """

    def __init__(self, *, index_dir: Path, models: Path = Path("models"), threads: int = 2):
        if not 1 <= threads <= 8:
            raise ValueError("Use 1–8 inference threads")
        self.root, self.models, self.threads = index_dir, models, threads
        self.encoder: ResidentQueryEncoder | None = None
        self.encoders: dict[str, ResidentQueryEncoder] = {}
        self.precision: str | None = None
        self.closed = False
        self._operation_lock = RLock()

    def _select(self, precision: str):
        if self.closed:
            raise RuntimeError("Library session is closed")
        if self.encoder is None or self.precision != precision:
            if self.encoder is not None:
                self.encoder.release()
            if precision not in self.encoders:
                self.encoders[precision] = ResidentQueryEncoder()
            self.encoder = self.encoders[precision]
            self.precision = precision
        self.encoder.verify(self.models, self.threads, precision)

    def close(self):
        with self._operation_lock:
            for encoder in self.encoders.values():
                encoder.close()
            self.encoders.clear()
            self.encoder = None
            self.closed = True

    def search(self, query: str, **options: Any) -> dict:
        """Serialize owned encoder lifetime, including mixed-precision switches."""
        if not self._operation_lock.acquire(timeout=120):
            raise TimeoutError("Library search session is busy")
        try:
            return self._search(query, **options)
        finally:
            self._operation_lock.release()

    def _search(
        self,
        query: str,
        *,
        limit: int = 50,
        mode: str = "hybrid",
        use_feedback: bool = True,
        save_run: bool = False,
        on_progress: Callable[[dict], None] | None = None,
        trace: dict | None = None,
        profile: dict | None = None,
        hidden_only: bool = False,
        offset: int = 0,
    ) -> dict:
        if self.closed:
            raise RuntimeError("Library session is closed")
        if hidden_only and (mode != "lexical" or save_run or not use_feedback):
            raise ValueError("Hidden browsing requires unrecorded lexical feedback mode")
        if offset < 0:
            raise ValueError("Offset must be nonnegative")
        if (
            not query.strip()
            or len(query) > 4096
            or not 1 <= limit <= 100
            or mode not in {"hybrid", "visual", "lexical", "semantic"}
        ):
            raise ValueError("Use a valid query, search mode and 1–100 results")
        started = perf_counter()
        diagnostics = trace if trace is not None else {}
        diagnostics.clear()
        if profile is not None:
            profile.clear()

        def progress(phase, **extra):
            if on_progress:
                on_progress({"phase": phase, **extra})

        progress("library_snapshots")
        encoder_source = find_spec("realopen_lens.encoders.onnx")
        if encoder_source is None or encoder_source.origin is None:
            raise ValueError("Query encoder source provenance is unavailable")
        encoder_hash = digest(Path(encoder_source.origin)) if mode != "lexical" else None
        catalog = Catalog(self.root, readonly=True)
        try:
            # Capture all generation pointers in one short read transaction. Release
            # before inference; immutable artifacts remain available after refresh.
            catalog.db.execute("BEGIN")
            folders = [dict(r) for r in catalog.db.execute("SELECT * FROM collections ORDER BY id")]
            if len(folders) > MAX_FOLDERS:
                raise ValueError("Library search supports at most 256 folders")
            snapshots = []
            pending = []
            count = 0
            for folder in folders:
                if not folder["active_generation"]:
                    pending.append(folder["name"])
                    continue
                generation = dict(
                    catalog.db.execute(
                        "SELECT * FROM generations WHERE id=? AND collection_id=?",
                        (folder["active_generation"], folder["id"]),
                    ).fetchone()
                )
                manifest = load_manifest(catalog, generation)
                count += len(manifest["images"])
                if count > MAX_IMAGES:
                    raise ValueError(
                        "Library search supports at most 100,000 indexed image entries"
                    )
                spec = manifest["config"]["spec"]
                precision = spec.get("siglip_precision", "int8")
                if precision not in {"int8", "fp32"}:
                    raise ValueError(f"Unsupported precision in folder {folder['name']}")
                if mode != "lexical":
                    if (
                        spec["model_registry"] != registry()
                        or spec["inference_sha256"] != encoder_hash
                        or any(
                            spec["packages"][n] != version(n)
                            for n in ("numpy", "onnxruntime", "tokenizers")
                        )
                    ):
                        raise ValueError(f"Re-index incompatible folder {folder['name']!r}")
                snapshots.append((folder, generation, manifest, precision))
            catalog.db.commit()
            paths = {}
            pixels = {}
            unavailable = []
            for position, (folder, generation, manifest, precision) in enumerate(snapshots):
                progress("library_source_validation", step=position, max_steps=len(snapshots))
                valid = source_paths(Path(folder["root"]), manifest)
                paths[folder["id"]] = valid
                if len(valid) < len(manifest["images"]):
                    unavailable.append(
                        {
                            "collection": folder["name"],
                            "images": len(manifest["images"]) - len(valid),
                        }
                    )
                cached = duplicates.snapshot(self.root, manifest)
                if cached:
                    pixels.update(cached.get("pixels", {}))
            # Canonical decoded-pixel groups when published scans exist; bytes
            # otherwise. Never decode/hash source images while searching.
            groups: dict[str, set[str]] = {}
            for valid in paths.values():
                for key in valid:
                    groups.setdefault(
                        "pixels:" + pixels[key] if key in pixels else "bytes:" + key, set()
                    ).add(key)
            canonical = {key: min(ids) for ids in groups.values() for key in ids}
            selected: dict[str, tuple[tuple, dict, dict, str]] = {}
            sources: dict[str, list[dict]] = {}
            records = {}
            for folder, generation, manifest, precision in snapshots:
                valid = paths[folder["id"]]
                texts = {r["image_id"]: r for r in manifest["ocr"]}
                for row in manifest["images"]:
                    key = row["image_id"]
                    if key not in valid:
                        continue
                    group = canonical[key]
                    sources.setdefault(group, []).append(
                        {
                            "collection": folder["name"],
                            "collection_id": folder["id"],
                            "generation": generation["id"],
                            "image_id": key,
                            "paths": valid[key],
                            "siglip_precision": precision,
                        }
                    )
                    # Deterministic preferred representation: FP32, then stable
                    # collection/image IDs. Duplicates do not get multiple votes.
                    preference = (precision != "fp32", folder["id"], key)
                    if group not in selected or preference < selected[group][0]:
                        selected[group] = (preference, folder, manifest, key)
                        records[group] = {**texts[key], "image_id": group}
            visual = {}
            semantic = {}
            semantic_query = None
            ordered = sorted(snapshots, key=lambda s: (s[3] != self.precision, s[3], s[0]["id"]))
            visual_queries = {}
            if mode != "lexical":
                for position, (folder, generation, manifest, precision) in enumerate(ordered):
                    valid = paths[folder["id"]]
                    if not valid:
                        continue
                    progress(
                        "library_encoding",
                        step=position,
                        max_steps=len(ordered),
                        collection=folder["name"],
                        siglip_precision=precision,
                    )
                    if (
                        mode in {"hybrid", "visual"}
                        and precision not in visual_queries
                        or mode in {"hybrid", "semantic"}
                        and semantic_query is None
                    ):
                        self._select(precision)
                    assert self.encoder is not None
                    if mode in {"hybrid", "semantic"} and semantic_query is None:
                        semantic_query = self.encoder.encode(
                            query, "e5", self.models, self.threads, precision
                        )
                    if mode in {"hybrid", "visual"} and precision not in visual_queries:
                        visual_queries[precision] = self.encoder.encode(
                            query, "siglip2", self.models, self.threads, precision
                        )
                    progress(
                        "library_retrieval",
                        step=position,
                        max_steps=len(ordered),
                        collection=folder["name"],
                    )
                    vision, passages = matrices(catalog, manifest)
                    vs = (
                        ranking.vector_scores(
                            vision, visual_queries[precision], manifest["vision_ids"]
                        )
                        if mode in {"hybrid", "visual"}
                        else {}
                    )
                    ss = (
                        ranking.vector_scores(passages, semantic_query, manifest["passage_ids"])
                        if semantic_query is not None
                        else {}
                    )
                    for key in valid:
                        group = canonical[key]
                        chosen = selected[group]
                        if chosen[1]["id"] == folder["id"] and chosen[3] == key:
                            if key in vs:
                                visual[group] = vs[key]
                            if key in ss:
                                semantic[group] = ss[key]
                    del vision, passages
            progress("library_fusion")
            ocr = list(records.values())
            depths = CandidateDepths(visual=max(50, limit))
            index = ranking.lexical_index(ocr)
            try:
                lexical = ranking.rank_lexical(index, query, max(len(selected), 1))
                if mode == "hybrid":
                    ranked = ranking.hybrid_scores(
                        visual,
                        semantic,
                        query,
                        ocr,
                        limit=max(len(selected), 1),
                        lexical_candidates=lexical,
                        lexical_statistics=ranking.lexical_frequencies(index, query),
                        candidate_depths=depths,
                        trace=diagnostics,
                    )
                elif mode == "lexical":
                    ranked = lexical
                else:
                    scores = visual if mode == "visual" else semantic
                    ranked = sorted(scores, key=lambda k: (-scores[k], k))
            finally:
                index.close()
            # Library exact-query memory has its own namespace. A folder label
            # cannot hide copies elsewhere; library decisions cover the whole group.
            synthetic = {"pipeline": LIBRARY_ID, "images": []}
            decisions: dict[str, dict] = {}
            feedback: dict = {"enabled": False, "applied": False}
            if use_feedback:
                decisions, feedback = memory.resolve(
                    self.root, LIBRARY_ID, query, "", synthetic, canonical, content_pixels=pixels
                )
                latest: dict[str, dict] = {}
                for key, decision in decisions.items():
                    group = canonical[key]
                    if decision["event_id"] > latest.get(group, {}).get("event_id", 0):
                        latest[group] = decision
                decisions = latest
                ranked = memory.apply(ranked, decisions)
            if hidden_only:
                ranked = sorted(
                    (k for k, d in decisions.items() if d["label"] == "irrelevant"),
                    key=lambda k: (-decisions[k]["event_id"], k),
                )
            results: list[SearchResult] = []
            for group in ranked[offset : offset + limit]:
                progress(
                    "library_result_validation",
                    step=len(results),
                    max_steps=min(limit, len(ranked)),
                )
                _, folder, manifest, key = selected[group]
                row = search_results(catalog, manifest, paths[folder["id"]], [key], visual)[0]
                locations = sources[group]
                results.append(
                    replace(
                        row,
                        rank=len(results) + 1,
                        image_id=group,
                        paths=tuple(dict.fromkeys(p for loc in locations for p in loc["paths"])),
                        duplicate_ids=tuple(
                            sorted(k for k, g in canonical.items() if g == group and k != group)
                        ),
                        visual_similarity=visual.get(group),
                        feedback=decisions.get(group),
                        sources=tuple(locations),
                    )
                )
            provenance = [
                {
                    "collection": f["name"],
                    "collection_id": f["id"],
                    "generation": g["id"],
                    "manifest_sha256": g["checksum"],
                    "pipeline": m["pipeline"],
                    "encoder_spec": m["config"]["spec"],
                }
                for f, g, m, p in snapshots
            ]
            precisions = {p for f, g, m, p in snapshots if paths[f["id"]]}
            experimental = mode in {"hybrid", "visual"} and len(precisions) > 1
            diagnostics.update(
                mixed_precision_experimental=experimental,
                sources=provenance,
                searchable_images=len(selected),
                unavailable=unavailable,
                pending_folders=pending,
                merge_policy="global-cosine-hybrid-v1",
            )
            run_id = None
            if save_run:
                from realopen_lens.storage.feedback import FeedbackStore

                payload = dict(
                    version=1,
                    query={"kind": "text", "text": query},
                    collection="All folders",
                    collection_id=LIBRARY_ID,
                    generation=None,
                    manifest_sha256=None,
                    pipeline=LIBRARY_ID,
                    encoder_spec={},
                    library_sources=provenance,
                    ranking={
                        "mixed_precision_experimental": experimental,
                        "mode": mode,
                        "limit": limit,
                        "siglip_precision": (
                            next(iter(precisions)) if len(precisions) == 1 else "mixed"
                        ),
                        "reranker_sha256": None,
                        "feedback_memory": feedback,
                        "merge_policy": "global-cosine-hybrid-v1",
                    },
                    ranking_sources={"library.py": digest(Path(__file__))},
                    candidate_schema=None,
                    candidates={},
                    results=[asdict(r) for r in results],
                )
                with FeedbackStore(self.root) as store:
                    run_id = store.add_run(payload)
            if profile is not None:
                profile.update(
                    total_seconds=perf_counter() - started,
                    visual_encodings=len(visual_queries),
                    semantic_encodings=int(semantic_query is not None),
                    searchable_images=len(selected),
                    folders=len(snapshots),
                )
            return {
                "results": [asdict(r) for r in results],
                "run_id": run_id,
                "ranking": {
                    "kind": mode,
                    "sha256": None,
                    "mixed_precision_experimental": experimental,
                },
                "feedback": feedback,
                "unavailable": unavailable,
                "pending_folders": pending,
                "total": len(ranked),
                "offset": offset,
                "generation": None,
            }
        finally:
            catalog.close()
