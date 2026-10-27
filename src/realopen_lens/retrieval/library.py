"""Global channel ranking over immutable folder snapshots; no per-folder quotas."""

from collections.abc import Callable
from dataclasses import asdict, replace
from importlib.metadata import version
from importlib.util import find_spec
import hashlib
import json
from pathlib import Path
from time import perf_counter
from threading import RLock
from typing import Any

import numpy as np

from realopen_lens.indexing.stage_cache import encoder_identity, normalize_encoder
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

    def browse(self, **options: Any) -> dict:
        """Browse library collection membership without model inference."""
        return self.search(
            "",
            mode="browse",
            use_feedback=False,
            expected_generation=options.pop("generation", None),
            **options,
        )

    def similar(
        self,
        image: Path | None = None,
        *,
        image_id: str | None = None,
        collection: str = "personal",
        include_self: bool = False,
        **options: Any,
    ) -> dict:
        """Search every folder using compatible image vectors, never text embeddings."""
        if (image is None) == (image_id is None):
            raise ValueError("Supply exactly one image path or image_id")
        if not self._operation_lock.acquire(timeout=120):
            raise TimeoutError("Library search session is busy")
        try:
            return self._search(
                "",
                mode="visual",
                use_feedback=False,
                image=image,
                image_id=image_id,
                image_collection=collection,
                include_self=include_self,
                **options,
            )
        finally:
            self._operation_lock.release()

    def _search(
        self,
        query: str,
        *,
        limit: int = 50,
        deep_search: bool = False,
        mode: str = "hybrid",
        use_feedback: bool = True,
        interpret_metadata: bool = True,
        save_run: bool = False,
        defer_record: bool = False,
        on_progress: Callable[[dict], None] | None = None,
        trace: dict | None = None,
        profile: dict | None = None,
        hidden_only: bool = False,
        offset: int = 0,
        image: Path | None = None,
        image_id: str | None = None,
        image_collection: str = "personal",
        include_self: bool = False,
        virtual_collection: str | None = None,
        virtual_revision: int | None = None,
        expected_generation: str | None = None,
        browse_order: str = "newest",
        expected_snapshot: str | None = None,
    ) -> dict:
        from realopen_lens.organization.collections import resolve_scope, ensure_scope
        from realopen_lens.organization.library_collections import match_members
        from realopen_lens.storage.organization import CollectionConflict

        from realopen_lens.metadata import timeline

        if mode == "browse" and browse_order not in timeline.ORDERS:
            raise ValueError("Use newest or oldest timeline order")
        scope = None
        members: set[str] = set()
        if virtual_collection:
            members, scope = resolve_scope(
                self.root, LIBRARY_ID, virtual_collection, virtual_revision
            )
        elif virtual_revision is not None:
            raise ValueError("A virtual revision requires a virtual collection")
        image_query = image is not None or image_id is not None
        if self.closed:
            raise RuntimeError("Library session is closed")
        if hidden_only and (mode not in {"lexical", "browse"} or save_run or not use_feedback):
            raise ValueError("Hidden browsing requires unrecorded lexical feedback mode")
        if not 0 <= offset <= 1_000_000:
            raise ValueError("Offset must be nonnegative")
        if (
            (not image_query and mode != "browse" and not query.strip())
            or len(query) > 4096
            or not 1 <= limit <= (1000 if deep_search else 100)
            or mode not in {"hybrid", "visual", "lexical", "semantic", "browse"}
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
        encoder_hash = (
            encoder_identity(Path(encoder_source.origin))
            if mode not in {"lexical", "browse"}
            else None
        )
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
                if mode not in {"lexical", "browse"}:
                    if (
                        spec["model_registry"] != registry()
                        or normalize_encoder(spec["inference_sha256"])
                        not in {encoder_hash, digest(Path(encoder_source.origin))}
                        or any(
                            spec["packages"][n] != version(n)
                            for n in (
                                ("numpy", "Pillow", "onnxruntime")
                                if image_query
                                else ("numpy", "onnxruntime", "tokenizers")
                            )
                        )
                    ):
                        raise ValueError(f"Re-index incompatible folder {folder['name']!r}")
                snapshots.append((folder, generation, manifest, precision))
            catalog.db.commit()
            # Image inference is disposable; release any resident text model first.
            query_vectors = {}
            query_path = None
            query_pixels = None
            if image_query:
                if self.encoder is not None:
                    self.encoder.release()
                if image is not None:
                    query_path = image.expanduser().resolve(strict=True)
                    if not query_path.is_file() or query_path.stat().st_size > 64 * 1024 * 1024:
                        raise ValueError("Query image must be a local file no larger than 64 MiB")
                    image_id = digest(query_path)
                else:
                    source = next((s for s in snapshots if s[0]["name"] == image_collection), None)
                    if source is None or image_id not in source[2]["vision_ids"]:
                        raise ValueError("Query image is not in the source folder generation")
                    valid_query = source_paths(Path(source[0]["root"]), source[2])
                    if image_id not in valid_query:
                        raise ValueError("Query image source is unavailable or changed")
                    query_path = Path(valid_query[image_id][0])
                from realopen_lens.organization.fingerprints import fingerprint

                try:
                    query_pixels = fingerprint(query_path)["pixels"]
                except (OSError, ValueError):
                    pass  # Same decoded-pixel eligibility as duplicate scans.
                if digest(query_path) != image_id:
                    raise ValueError("Query image changed during duplicate validation")
                for f, g, m, precision in snapshots:
                    if image_id in m["vision_ids"] and precision not in query_vectors:
                        vision, passages = matrices(catalog, m)
                        query_vectors[precision] = np.array(
                            vision[m["vision_ids"].index(image_id)], dtype=np.float32, copy=True
                        )
                        del vision, passages
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
            missing_members = 0
            if scope is not None:
                allowed, missing_members = match_members(self.root, members, canonical, pixels)
                paths = {
                    folder: {k: v for k, v in valid.items() if canonical[k] in allowed}
                    for folder, valid in paths.items()
                }
            from realopen_lens.retrieval.metadata import compile_query, filter_paths, QueryPlan
            from realopen_lens.storage.feedback import memory_scope

            progress("interpreting_metadata")
            plan = (
                compile_query(
                    query,
                    [
                        row
                        for folder, gen, manifest, precision in snapshots
                        for row in manifest["images"]
                        if row["image_id"] in paths[folder["id"]]
                    ],
                )
                if interpret_metadata and query
                else QueryPlan(query, query)
            )
            interpretation = plan.report() if plan.constraints or plan.notices else None
            search_query = plan.text
            metadata_only = bool(plan.constraints) and not search_query
            retrieval_mode = "browse" if metadata_only else mode
            paths = {
                folder["id"]: filter_paths(plan, manifest, paths[folder["id"]], folder["name"])
                for folder, gen, manifest, precision in snapshots
            }
            if interpretation:
                diagnostics["query_interpretation"] = interpretation
            snapshot_token = hashlib.sha256(
                json.dumps(
                    [(f["id"], g["id"], sorted(paths[f["id"]])) for f, g, m, p in snapshots]
                ).encode()
            ).hexdigest()
            if expected_generation is not None and expected_generation != snapshot_token:
                raise CollectionConflict("Library index changed; refresh the virtual collection")
            selected: dict[str, tuple[tuple, dict, dict, str]] = {}
            sources: dict[str, list[dict]] = {}
            marker_lookup: dict[str, dict] = {}
            date_options: dict[str, list[dict]] = {}
            records = {}
            for folder, generation, manifest, precision in snapshots:
                valid = paths[folder["id"]]
                texts = {r["image_id"]: r for r in manifest["ocr"]}
                for row in manifest["images"]:
                    key = row["image_id"]
                    if key not in valid:
                        continue
                    group = canonical[key]
                    if mode == "browse":
                        date_options.setdefault(group, []).append(
                            timeline.image_date(row, valid[key], Path(folder["root"]))
                        )
                    from realopen_lens.retrieval.metadata import marked_screenshot

                    if group not in marker_lookup or marked_screenshot(row):
                        marker_lookup[group] = row
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
            dates = {key: timeline.preferred_date(values) for key, values in date_options.items()}
            timeline_token = (
                timeline.snapshot([snapshot_token, scope], dates, browse_order, expected_snapshot)
                if mode == "browse"
                else None
            )
            visual = {}
            semantic = {}
            semantic_query = None
            ordered = sorted(snapshots, key=lambda s: (s[3] != self.precision, s[3], s[0]["id"]))
            visual_queries = {}
            if retrieval_mode not in {"lexical", "browse"}:
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
                    if not image_query and (
                        retrieval_mode in {"hybrid", "visual"}
                        and precision not in visual_queries
                        or retrieval_mode in {"hybrid", "semantic"}
                        and semantic_query is None
                    ):
                        self._select(precision)
                    if image_query and precision not in visual_queries:
                        if precision in query_vectors:
                            vector = query_vectors[precision]
                        else:
                            from realopen_lens.retrieval.images import (
                                image_vector,
                                verify_image_encoder,
                            )

                            verify_image_encoder(manifest["config"]["spec"], self.models, precision)
                            assert query_path is not None
                            vector = image_vector(query_path, self.models, self.threads, precision)
                        norm = np.linalg.norm(vector)
                        if vector.shape != (768,) or not np.isfinite(vector).all() or norm == 0:
                            raise ValueError("Invalid image query embedding")
                        visual_queries[precision] = vector / norm
                    if (
                        not image_query
                        and retrieval_mode in {"hybrid", "semantic"}
                        and semantic_query is None
                    ):
                        assert self.encoder is not None
                        semantic_query = self.encoder.encode(
                            search_query, "e5", self.models, self.threads, precision
                        )
                    if (
                        not image_query
                        and retrieval_mode in {"hybrid", "visual"}
                        and precision not in visual_queries
                    ):
                        assert self.encoder is not None
                        visual_queries[precision] = self.encoder.encode(
                            search_query, "siglip2", self.models, self.threads, precision
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
                        if retrieval_mode in {"hybrid", "visual"}
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
            depths = (
                CandidateDepths(1000, 1000, 1000)
                if deep_search
                else CandidateDepths(visual=max(50, limit))
            )
            index = ranking.lexical_index(ocr) if retrieval_mode != "browse" else None
            try:
                lexical = (
                    ranking.rank_lexical(index, search_query, max(len(selected), 1))
                    if index is not None
                    else []
                )
                if retrieval_mode == "browse":
                    ranked = (
                        timeline.ordered(dates, browse_order)
                        if mode == "browse"
                        else sorted(selected)
                    )
                elif retrieval_mode == "hybrid":
                    assert index is not None
                    ranked = ranking.hybrid_scores(
                        visual,
                        semantic,
                        search_query,
                        ocr,
                        limit=max(len(selected), 1),
                        lexical_candidates=lexical,
                        lexical_statistics=ranking.lexical_frequencies(index, search_query),
                        candidate_depths=depths,
                        trace=diagnostics,
                    )
                elif retrieval_mode == "lexical":
                    ranked = lexical
                else:
                    scores = visual if retrieval_mode == "visual" else semantic
                    ranked = sorted(scores, key=lambda k: (-scores[k], k))
            finally:
                if index is not None:
                    index.close()
            if image_query:
                assert query_path is not None
                if digest(query_path) != image_id:
                    raise ValueError("Query image changed during search; retry with a stable file")
                if not include_self:
                    excluded = {canonical[image_id]} if image_id in canonical else set()
                    if query_pixels is not None:
                        excluded.update(
                            canonical[k]
                            for k, p in pixels.items()
                            if p == query_pixels and k in canonical
                        )
                    ranked = [k for k in ranked if k not in excluded]
            from realopen_lens.retrieval.metadata import screenshot_ranking

            ranked = screenshot_ranking(plan, ranked, marker_lookup, visual)
            # Library exact-query memory has its own namespace. A folder label
            # cannot hide copies elsewhere; library decisions cover the whole group.
            synthetic = {"pipeline": LIBRARY_ID, "images": []}
            decisions: dict[str, dict] = {}
            feedback: dict = {"enabled": False, "applied": False}
            if use_feedback:
                decisions, feedback = memory.resolve(
                    self.root,
                    LIBRARY_ID,
                    query,
                    memory_scope((scope or {}).get("id", ""), interpretation),
                    synthetic,
                    {key: group for key, group in canonical.items() if group in selected},
                    content_pixels=pixels,
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
                query_image_id=image_id,
                reused_image_precisions=sorted(query_vectors),
                merge_policy="global-cosine-hybrid-v1",
            )
            if scope is not None:
                ensure_scope(self.root, LIBRARY_ID, scope)
            run_id = None
            record_payload = None
            if save_run:
                from realopen_lens.storage.feedback import FeedbackStore

                payload = dict(
                    version=1,
                    query=(
                        {"kind": "image", "image_id": image_id}
                        if image_query
                        else {"kind": "text", "text": query}
                    ),
                    collection="All folders",
                    collection_id=LIBRARY_ID,
                    generation=None,
                    manifest_sha256=None,
                    pipeline=LIBRARY_ID,
                    encoder_spec={},
                    library_sources=provenance,
                    scope=scope,
                    ranking={
                        "mixed_precision_experimental": experimental,
                        "mode": mode,
                        "query_interpretation": interpretation,
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
                if defer_record:
                    record_payload = payload
                else:
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
                **(
                    {
                        **timeline.page([asdict(r) for r in results], dates, offset, len(ranked)),
                        "snapshot": timeline_token,
                        "order": browse_order,
                    }
                    if mode == "browse"
                    else {"results": [asdict(r) for r in results]}
                ),
                "run_id": run_id,
                **({"_record_payload": record_payload} if defer_record else {}),
                "interpretation": interpretation,
                "ranking": {
                    "kind": "metadata" if metadata_only else mode,
                    "sha256": None,
                    "mixed_precision_experimental": experimental,
                },
                "feedback": feedback,
                "unavailable": unavailable,
                "pending_folders": pending,
                "total": len(ranked),
                "offset": offset,
                "generation": snapshot_token if mode == "browse" else None,
                "scope": scope,
                "member_count": len(members) if scope is not None else len(selected),
                "unavailable_count": missing_members,
            }
        finally:
            catalog.close()
