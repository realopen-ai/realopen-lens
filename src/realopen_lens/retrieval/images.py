"""Visual similarity queries against immutable saved collection generations."""

from importlib.metadata import version
from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from realopen_lens.indexing.stage_cache import encoder_identity, normalize_encoder
from realopen_lens.models.manager import artifact_path, digest, registry
from realopen_lens.retrieval.ranking import vector_scores
from realopen_lens.retrieval.search import SearchResult, search_results, source_paths
from realopen_lens.runtime.jobs import environment
from realopen_lens.storage.artifacts import load_manifest, matrices
from realopen_lens.storage.catalog import Catalog, default_directory


def verify_image_encoder(spec: dict, models: Path, precision: str) -> None:
    """Check preprocessing compatibility and only the required vision graph."""
    from realopen_lens.encoders import onnx as inference

    if spec["model_registry"] != registry() or normalize_encoder(spec["inference_sha256"]) not in {
        encoder_identity(Path(inference.__file__)),
        digest(Path(inference.__file__)),
    }:
        raise ValueError("Index model/preprocessing version differs; run 'lens index' again")
    for name in ("numpy", "Pillow", "onnxruntime"):
        if spec["packages"][name] != version(name):
            raise ValueError("Image encoder dependencies changed; run 'lens index' again")
    key = f"vision_{precision}"
    path = artifact_path(models, "siglip2", key)
    expected = registry()["models"]["siglip2"]["files"][key]
    if not path.is_file():
        raise ValueError(f"Missing model file: {path}. Run 'lens models fetch' first.")
    if path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
        raise ValueError(f"Model integrity check failed: {path}")


def image_vector(image: Path, models: Path, threads: int, precision: str) -> np.ndarray:
    """Encode one local image in a bounded, offline, disposable worker."""
    request = {
        "image": str(image),
        "models": str(models.resolve()),
        "threads": threads,
        "siglip_precision": precision,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "realopen_lens.runtime.worker", "--image-query"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            env=environment(threads),
            timeout=120,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("Image query encoder exceeded its deadline") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Local image encoder failed: " + error.stderr[-1500:]) from error
    vector = np.asarray(json.loads(result.stdout), dtype=np.float32)
    if vector.shape != (768,) or not np.isfinite(vector).all():
        raise ValueError("Invalid image query embedding")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid image query embedding norm")
    return vector / norm


def search_similar(
    image: Path | None = None,
    *,
    image_id: str | None = None,
    collection: str = "personal",
    virtual_collection: str | None = None,
    virtual_revision: int | None = None,
    index_dir: Path | None = None,
    models: Path = Path("models"),
    limit: int = 10,
    threads: int = 2,
    include_self: bool = False,
    siglip_precision: str | None = None,
    trace: dict | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> list[SearchResult]:
    """Find visually similar images using a local path OR a published image ID.

    Exact byte matches reuse stored vectors, even for copies outside the source
    folder. Indexed-ID queries need no models or original query file. Result
    sources must still exist unchanged. Scores are cosine similarities, not
    relevance probabilities or near-duplicate decisions.
    """
    if (image is None) == (image_id is None):
        raise ValueError("Supply exactly one image path or image_id")
    if not 1 <= limit <= 100 or not 1 <= threads <= 8:
        raise ValueError("Use 1–100 results and 1–8 threads")
    if siglip_precision is not None and siglip_precision not in {"int8", "fp32"}:
        raise ValueError("SigLIP2 precision must be int8 or fp32")
    if trace is not None:
        trace.clear()
    from realopen_lens.indexing.service import signature

    query_signature = None
    if image is not None:
        image = image.expanduser().resolve(strict=True)
        if not image.is_file():
            raise ValueError("Query image must be a local file")
        query_signature = signature(image)
        if query_signature["size"] > 64 * 1024 * 1024:
            raise ValueError("Query image exceeds 64 MiB encoded size")
        image_id = digest(image)
    if on_progress is not None:
        on_progress({"phase": "index_verification"})
    catalog = Catalog(index_dir or default_directory(), readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        manifest = load_manifest(catalog, generation)
        precision = manifest["config"]["spec"].get("siglip_precision", "int8")
        if precision not in {"int8", "fp32"}:
            raise ValueError("Unsupported collection SigLIP2 precision")
        if siglip_precision is not None and siglip_precision != precision:
            raise ValueError("Requested SigLIP2 precision differs from this collection")
        scope = None
        if virtual_collection is not None:
            from realopen_lens.organization.collections import resolve_scope, ensure_scope

            allowed, scope = resolve_scope(
                catalog.root, found["id"], virtual_collection, virtual_revision
            )
        elif virtual_revision is not None:
            raise ValueError("Virtual revision requires a virtual collection")
        visual, _ = matrices(catalog, manifest)
        ids = manifest["vision_ids"]
        reused = image_id in ids
        if reused:
            query = np.array(visual[ids.index(image_id)], copy=True)
        elif image is None:
            raise ValueError("Image ID is not in this collection's active generation")
        else:
            if on_progress is not None:
                on_progress({"phase": "model_verification"})
            verify_image_encoder(manifest["config"]["spec"], models, precision)
            if on_progress is not None:
                on_progress({"phase": "encoding_image"})
            query = image_vector(image, models, threads, precision)
        if image is not None:
            assert query_signature is not None
            after = signature(image)
            if after != query_signature:
                # Reading a new file can trigger metadata-only ctime changes on
                # some filesystems. Confirm its bytes rather than rejecting it.
                stable = ("size", "mtime_ns", "device", "inode")
                if (
                    any(after[key] != query_signature[key] for key in stable)
                    or digest(image) != image_id
                    or any(signature(image)[key] != after[key] for key in stable)
                ):
                    raise ValueError("Query image changed during search; retry with a stable file")
        norm = np.linalg.norm(query)
        if query.shape != (768,) or not np.isfinite(norm) or norm == 0:
            raise ValueError("Invalid image query embedding")
        query = query / norm
        if on_progress is not None:
            on_progress({"phase": "retrieval"})
        paths = source_paths(Path(found["root"]), manifest)
        all_paths = paths
        if scope is not None:
            paths = {key: value for key, value in paths.items() if key in allowed}
        scores = vector_scores(visual, query, ids)
        from realopen_lens.organization.duplicates import exact_groups, snapshot, collapse

        groups = exact_groups(catalog.root, manifest, all_paths)
        excluded = (
            set(groups.get(image_id, (image_id,)))
            if not include_self and image_id is not None
            else set()
        )
        if not include_self:
            duplicate_snapshot = snapshot(catalog.root, manifest)
            if duplicate_snapshot is not None:
                pixels = duplicate_snapshot.get("pixels", {})
                query_pixels = pixels.get(image_id)
                if query_pixels is None and image is not None:
                    from realopen_lens.organization.fingerprints import fingerprint

                    try:
                        query_pixels = fingerprint(image)["pixels"]
                    except (OSError, ValueError):
                        query_pixels = None  # Same multi-frame/size eligibility as the scan.
                    if digest(image) != image_id:
                        raise ValueError("Query image changed during duplicate validation")
                if query_pixels is not None:
                    excluded.update(key for key, value in pixels.items() if value == query_pixels)
        ranking = sorted(
            (key for key in scores if key in paths and key not in excluded),
            key=lambda key: (-scores[key], key),
        )
        if trace is not None:
            trace.update(
                generation=generation["id"],
                manifest_sha256=generation["checksum"],
                siglip_precision=precision,
                query_image_id=image_id,
                reused_embedding=reused,
                searchable_images=len(paths),
                include_self=include_self,
            )
        ranking = collapse(ranking, groups)[:limit]
        results = search_results(catalog, manifest, paths, ranking, scores, deduplicate=True)
        if scope is not None:
            ensure_scope(catalog.root, found["id"], scope)
            if trace is not None:
                trace["scope"] = scope
        return results
    finally:
        catalog.close()
