"""Search a published collection with existing visual, lexical, and hybrid ranking."""

from dataclasses import dataclass, replace
from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from realopen_lens.storage.catalog import Catalog, default_directory, local_path
from realopen_lens.storage.artifacts import load_manifest, matrices
from realopen_lens.runtime.jobs import environment
from realopen_lens.runtime.resident import ResidentQueryEncoder
from realopen_lens.indexing.stage_cache import encoder_identity, normalize_encoder
from realopen_lens.models.manager import digest, registry, verify
from realopen_lens.retrieval.candidates import CandidateDepths
from realopen_lens.retrieval.ranking import (
    hybrid_rank,
    lexical_index,
    open_lexical,
    lexical_scores,
    lexical_frequencies,
    rank_lexical,
    rank_vectors,
    vector_scores,
)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    image_id: str
    paths: tuple[str, ...]
    thumbnail: str | None
    width: int
    height: int
    text: str
    visual_similarity: float | None
    status: str = "ready"
    issues: tuple[dict, ...] = ()
    duplicate_ids: tuple[str, ...] = ()
    feedback: dict | None = None
    sources: tuple[dict, ...] = ()


def source_paths(root: Path, manifest: dict) -> dict[str, tuple[str, ...]]:
    """Return only unchanged local sources within the collection root."""
    from realopen_lens.indexing.service import signature

    paths_by_id = {}
    for row in manifest["images"]:
        paths = []
        for relative in row["paths"]:
            try:
                source = (root / relative).resolve()
                if (
                    source.is_relative_to(root)
                    and source.is_file()
                    and signature(source) == row["signatures"][relative]
                ):
                    paths.append(str(source))
            except (OSError, RuntimeError):
                continue
        if paths:
            paths_by_id[row["image_id"]] = tuple(paths)
    return paths_by_id


def search_results(
    catalog: Catalog,
    manifest: dict,
    paths_by_id: dict[str, tuple[str, ...]],
    ranking: list[str],
    scores: dict[str, float],
    *,
    deduplicate: bool = False,
) -> list[SearchResult]:
    """Build consistent metadata for text and image search results."""
    from realopen_lens.organization.duplicates import exact_groups, collapse

    groups = exact_groups(catalog.root, manifest, paths_by_id) if deduplicate else {}
    ranking = collapse(ranking, groups)
    lookup = {row["image_id"]: row for row in manifest["images"]}
    texts = {row["image_id"]: row["text"] for row in manifest["ocr"]}
    results: list[SearchResult] = []
    for image_id in ranking:
        row = lookup[image_id]
        members = (image_id, *(key for key in groups.get(image_id, ()) if key != image_id))
        result_paths = tuple(dict.fromkeys(p for key in members for p in paths_by_id[key]))
        thumbnail = local_path(catalog.root, row["thumbnail"])
        results.append(
            SearchResult(
                len(results) + 1,
                image_id,
                result_paths,
                (
                    str(thumbnail)
                    if thumbnail.is_file() and digest(thumbnail) == row["thumbnail_sha256"]
                    else None
                ),
                row["metadata"]["width"],
                row["metadata"]["height"],
                texts[image_id][:300],
                scores.get(image_id),
                row.get("status", "ready"),
                tuple(row.get("issues", [])),
                tuple(key for key in members if key != image_id),
            )
        )
    return results


def query_vector(
    text: str, kind: str, models: Path, threads: int, siglip_precision: str = "int8"
) -> np.ndarray:
    request = {"query": text, "kind": kind, "models": str(models.resolve()), "threads": threads}
    request["siglip_precision"] = siglip_precision
    try:
        result = subprocess.run(
            [sys.executable, "-m", "realopen_lens.runtime.worker", "--query"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            env=environment(threads),
            timeout=120,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("Query encoder exceeded its deadline") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Local query encoder failed: " + error.stderr[-1500:]) from error
    vector = np.asarray(json.loads(result.stdout), dtype=np.float32)
    if vector.shape != (768 if kind == "siglip2" else 384,) or not np.isfinite(vector).all():
        raise ValueError("Invalid query embedding")
    return vector


def search_collection(
    query: str,
    *,
    collection: str = "personal",
    virtual_collection: str | None = None,
    virtual_revision: int | None = None,
    index_dir: Path | None = None,
    models: Path = Path("models"),
    limit: int = 10,
    mode: str = "hybrid",
    threads: int = 2,
    trace: dict | None = None,
    reranker: Path | None = None,
    use_active_reranker: bool = True,
    use_feedback: bool = True,
    feedback_trace: dict | None = None,
    metadata_trace: dict | None = None,
    interpret_metadata: bool = True,
    candidate_depths: CandidateDepths | None = None,
    siglip_precision: str | None = None,
    query_encoder: ResidentQueryEncoder | None = None,
    profile: dict | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> list[SearchResult]:
    """Search the active generation; optional hybrid trace exposes bounded channel pools.

    Tracing preserves scores and ordering. Raw BM25 is lower-is-better, cosine
    similarities are higher-is-better, and absent channel values remain None.
    """
    explicit_reranker = reranker is not None
    if metadata_trace is not None:
        metadata_trace.clear()
    if feedback_trace is not None:
        feedback_trace.clear()
    started = checkpoint = perf_counter()
    if profile is not None:
        profile.clear()

    def stage(name: str):
        nonlocal checkpoint
        now = perf_counter()
        if profile is not None:
            profile[name] = now - checkpoint
        checkpoint = now

    if not query.strip() or len(query) > 4096 or not 1 <= limit <= 100 or not 1 <= threads <= 8:
        raise ValueError(
            "Use a nonempty query up to 4096 characters, 1–100 results, and 1–8 threads"
        )
    if mode not in {"hybrid", "visual", "lexical", "semantic"}:
        raise ValueError("Search mode must be hybrid, visual, lexical, or semantic")
    if candidate_depths is not None and mode != "hybrid":
        raise ValueError("Candidate depths require hybrid mode")
    if siglip_precision is not None and siglip_precision not in {"int8", "fp32"}:
        raise ValueError("SigLIP2 precision must be int8 or fp32")
    if virtual_collection is not None and reranker is not None:
        raise ValueError("Rerankers require unscoped searches until scoped evaluation is supported")
    if reranker is None and mode == "hybrid" and use_active_reranker and virtual_collection is None:
        from realopen_lens.learning.registry import active

        reranker = active(index_dir or default_directory(), collection)
    learned = None
    if reranker is not None:
        from realopen_lens.learning.logistic import load

        if mode != "hybrid":
            raise ValueError("The experimental reranker requires hybrid mode")
        learned = load(reranker)
        if candidate_depths is not None and candidate_depths != CandidateDepths():
            raise ValueError(
                "Reranker candidate budget differs; existing models require 50/channel"
            )
        if trace is None:
            trace = {}
    channel_limit = 50 if learned is not None else max(50, limit)
    depths = candidate_depths or CandidateDepths(visual=channel_limit)
    if mode == "hybrid":
        channel_limit = depths.lexical
    if trace is not None:
        trace.clear()
        if mode != "hybrid":
            raise ValueError("Candidate tracing requires hybrid mode")
    if on_progress is not None:
        on_progress({"phase": "index_verification"})
    catalog = Catalog(index_dir or default_directory(), readonly=True)
    index = None
    try:
        found, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
        stage("index_verification_seconds")
        precision = manifest["config"]["spec"].get("siglip_precision", "int8")
        if precision not in {"int8", "fp32"}:
            raise ValueError("Unsupported collection SigLIP2 precision")
        if siglip_precision is not None and siglip_precision != precision:
            raise ValueError("Requested SigLIP2 precision differs from this collection")
        if trace is not None:
            trace.update(
                generation=generation["id"],
                manifest_sha256=generation["checksum"],
                siglip_precision=precision,
            )
        if on_progress is not None:
            on_progress({"phase": "source_validation"})
        paths_by_id = source_paths(Path(found["root"]), manifest)
        scope = None
        if virtual_collection is not None:
            from realopen_lens.organization.collections import resolve_scope, ensure_scope

            allowed, scope = resolve_scope(
                catalog.root, found["id"], virtual_collection, virtual_revision
            )
            paths_by_id = {key: value for key, value in paths_by_id.items() if key in allowed}
            if trace is not None:
                trace["scope"] = scope
        elif virtual_revision is not None:
            raise ValueError("Virtual revision requires a virtual collection")
        from realopen_lens.retrieval.metadata import compile_query, filter_paths, QueryPlan
        from realopen_lens.storage.feedback import memory_scope

        if on_progress is not None:
            on_progress({"phase": "interpreting_metadata"})
        plan = (
            compile_query(
                query, [row for row in manifest["images"] if row["image_id"] in paths_by_id]
            )
            if interpret_metadata
            else QueryPlan(query, query)
        )
        interpretation = plan.report() if plan.constraints or plan.notices else None
        if metadata_trace is not None and interpretation:
            metadata_trace.update(interpretation)
        if trace is not None and interpretation:
            trace["query_interpretation"] = interpretation
        search_query = plan.text
        metadata_only = bool(plan.constraints) and not search_query
        paths_by_id = filter_paths(plan, manifest, paths_by_id, found["name"])
        if (
            any(
                rule["kind"] != "screenshot"
                or rule.get("policy") == "marked_only"
                or rule.get("promote")
                for rule in plan.constraints
            )
            and learned is not None
        ):
            if explicit_reranker:
                raise ValueError(
                    "Metadata-filtered queries require hybrid ranking until "
                    "filtered reranker evaluation is supported"
                )
            learned = None
        if learned is not None and precision != learned.compatibility.get(
            "siglip_precision", "int8"
        ):
            if "siglip_precision" not in learned.compatibility:
                raise ValueError("Existing reranker encoder compatibility requires INT8 SigLIP2")
            raise ValueError("Reranker SigLIP2 precision differs from this collection")
        if learned is not None and any(
            (
                normalize_encoder(manifest["config"]["spec"].get(key, ""))
                != normalize_encoder(value)
                if key == "inference_sha256"
                else manifest["config"]["spec"].get(key) != value
            )
            for key, value in learned.compatibility.items()
        ):
            raise ValueError("Reranker encoder compatibility differs from this collection")
        lookup = {
            row["image_id"]: row for row in manifest["images"] if row["image_id"] in paths_by_id
        }
        from realopen_lens.retrieval import memory

        decisions: dict[str, dict] = {}
        feedback_state = {"enabled": False, "policy": "exact-query-v1", "applied": False}
        if use_feedback:
            decisions, feedback_state = memory.resolve(
                catalog.root,
                found["id"],
                query,
                memory_scope(virtual_collection or "", interpretation),
                manifest,
                paths_by_id,
            )
        if feedback_trace is not None:
            feedback_trace.update(feedback_state)
        ocr_records = [r for r in manifest["ocr"] if r["image_id"] in lookup]
        stage("source_validation_seconds")
        if not lookup:
            if trace is not None:
                trace.update(
                    candidates=[],
                    rankings={key: [] for key in ("visual", "semantic", "lexical", "hybrid")},
                    searchable_images=0,
                    candidate_depths=depths.as_dict(),
                )
                if learned is not None:
                    trace["rankings"]["logistic"] = []
                    trace["reranker_sha256"] = learned.sha256
                    trace["reranker_scope"] = learned.scope
            if scope is not None:
                ensure_scope(catalog.root, found["id"], scope)
            return []
        visual_query = semantic_query = None
        if mode != "lexical" and not metadata_only:
            from realopen_lens.encoders import onnx as inference
            from importlib.metadata import version

            spec = manifest["config"]["spec"]
            if spec["model_registry"] != registry() or normalize_encoder(
                spec["inference_sha256"]
            ) not in {
                encoder_identity(Path(inference.__file__)),
                digest(Path(inference.__file__)),
            }:
                raise ValueError(
                    "Index model/preprocessing version differs; run 'lens index' again"
                )
            for name in ("numpy", "onnxruntime", "tokenizers"):
                if spec["packages"][name] != version(name):
                    raise ValueError("Query encoder dependencies changed; run 'lens index' again")
            if on_progress is not None:
                on_progress({"phase": "model_verification"})
            if query_encoder is None:
                verify(models, "int8", precision)
            else:
                query_encoder.verify(models, threads, precision)
            stage("model_verification_seconds")
            if mode in {"hybrid", "visual"}:
                if on_progress is not None:
                    on_progress({"phase": "encoding_visual_query"})
                visual_query = (
                    query_encoder.encode(search_query, "siglip2", models, threads, precision)
                    if query_encoder is not None
                    else (
                        query_vector(search_query, "siglip2", models, threads, precision)
                        if precision != "int8"
                        else query_vector(search_query, "siglip2", models, threads)
                    )
                )
                stage("visual_encoder_seconds")
            if mode in {"hybrid", "semantic"}:
                if on_progress is not None:
                    on_progress({"phase": "encoding_semantic_query"})
                semantic_query = (
                    query_encoder.encode(search_query, "e5", models, threads, precision)
                    if query_encoder is not None
                    else query_vector(search_query, "e5", models, threads)
                )
                stage("semantic_encoder_seconds")
        if on_progress is not None:
            on_progress({"phase": "retrieval"})
        path = local_path(catalog.root, manifest["artifacts"]["lexical"]["path"])
        index = (
            open_lexical(path, ocr_records)
            if len(lookup) == len(manifest["images"])
            else lexical_index(ocr_records)
        )
        lexical = (
            rank_lexical(index, search_query, limit=channel_limit) if not metadata_only else []
        )
        from realopen_lens.organization.duplicates import exact_groups, collapse

        duplicate_groups = exact_groups(catalog.root, manifest, paths_by_id)
        retrieval_limit = len(lookup) if duplicate_groups or feedback_state["applied"] else limit
        if (duplicate_groups or feedback_state["applied"]) and mode == "lexical":
            lexical = rank_lexical(index, search_query, limit=len(lookup))
        scores = {}
        if metadata_only:
            ranking = sorted(lookup)
            if trace is not None:
                trace.update(
                    candidates=[], rankings={"metadata": ranking}, candidate_depths=depths.as_dict()
                )
        elif mode == "lexical":
            ranking = lexical[:retrieval_limit]
        else:
            visual, semantic = matrices(catalog, manifest)
            vision_ids, passage_ids = manifest["vision_ids"], manifest["passage_ids"]
            if len(lookup) != len(manifest["images"]):
                visual = visual[[key in lookup for key in vision_ids]]
                semantic = semantic[[key in lookup for key in passage_ids]]
                vision_ids = [key for key in vision_ids if key in lookup]
                passage_ids = [key for key in passage_ids if key in lookup]
            if visual_query is not None:
                scores = vector_scores(visual, visual_query, vision_ids)
            if mode == "visual":
                assert visual_query is not None
                ranking = rank_vectors(visual, visual_query, vision_ids, retrieval_limit)
            elif mode == "semantic":
                assert semantic_query is not None
                ranking = (
                    rank_vectors(semantic, semantic_query, passage_ids, retrieval_limit)
                    if passage_ids
                    else []
                )
            else:
                assert visual_query is not None and semantic_query is not None
                ranking = hybrid_rank(
                    visual,
                    visual_query,
                    vision_ids,
                    semantic,
                    semantic_query,
                    passage_ids,
                    search_query,
                    ocr_records,
                    limit=50 if learned is not None else retrieval_limit,
                    lexical_candidates=lexical,
                    lexical_statistics=lexical_frequencies(index, search_query),
                    trace=trace,
                    candidate_depths=depths,
                )
        if trace is not None:
            bm25 = lexical_scores(index, search_query, limit=channel_limit)
            ranks = {
                mode: {key: i for i, key in enumerate(ranking, 1)}
                for mode, ranking in trace["rankings"].items()
            }
            texts_by_id = {row["image_id"]: row["text"] for row in ocr_records}
            for candidate in trace["candidates"]:
                key = candidate["image_id"]
                candidate.update(
                    lexical_bm25=bm25.get(key),
                    channel_ranks={name: values.get(key) for name, values in ranks.items()},
                    ocr_characters=len(texts_by_id.get(key, "")),
                    width=lookup[key]["metadata"]["width"],
                    height=lookup[key]["metadata"]["height"],
                    status=lookup[key].get("status", "ready"),
                )
            trace["searchable_images"] = len(lookup)
            if learned is not None:
                reranked = learned.rank(trace["candidates"])
                ranking = reranked[:retrieval_limit]
                trace["rankings"]["logistic"] = reranked[:50]
                trace["reranker_sha256"] = learned.sha256
                trace["reranker_scope"] = learned.scope
        stage("retrieval_seconds")
        if on_progress is not None:
            on_progress({"phase": "result_validation"})
        from realopen_lens.retrieval.metadata import screenshot_ranking

        ranking = screenshot_ranking(plan, ranking, lookup, scores)
        if trace is not None and any(rule.get("promote") for rule in plan.constraints):
            trace["rankings"]["screenshot_markers"] = ranking[:limit]
        if feedback_state["applied"]:
            ranking = memory.apply(ranking, decisions)
        ranking = collapse(ranking, duplicate_groups)[:limit]
        if trace is not None and feedback_state["applied"]:
            trace["feedback_memory"] = feedback_state
            trace["rankings"]["feedback"] = ranking
        results = search_results(catalog, manifest, paths_by_id, ranking, scores, deduplicate=True)
        results = [replace(row, feedback=decisions.get(row.image_id)) for row in results]
        if scope is not None:
            ensure_scope(catalog.root, found["id"], scope)
        stage("result_validation_seconds")
        return results
    finally:
        if index is not None:
            index.close()
        catalog.close()
        if profile is not None:
            profile["total_seconds"] = perf_counter() - started
