"""Loopback FastAPI adapter with authenticated local sessions and SSE jobs."""

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from realopen_lens.api import collections as reading
from realopen_lens.api.tasks import TaskManager, TERMINAL
from realopen_lens.indexing.service import collection_status
from realopen_lens.learning import feedback
from realopen_lens.runtime.jobs import writer_lock
from realopen_lens.storage.catalog import identifier


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: str = Field(default="personal", min_length=1, max_length=128)


class ScopedInput(Input):
    virtual_collection: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    virtual_revision: int | None = Field(default=None, ge=1)


class SearchInput(ScopedInput):
    query: str = Field(min_length=1, max_length=4096)
    mode: Literal["hybrid", "visual", "lexical", "semantic"] = "hybrid"
    limit: int = Field(default=50, ge=1, le=100)
    save_run: bool = False
    use_feedback: bool = True


class SimilarInput(ScopedInput):
    image_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    limit: int = Field(default=50, ge=1, le=100)
    save_run: bool = False


class IndexInput(Input):
    root: str = Field(min_length=1, max_length=4096)
    languages: str = Field(default="eng+fra+ara", pattern=r"^[a-zA-Z0-9_+.-]{1,128}$")
    siglip_precision: Literal["int8", "fp32"] = "int8"
    rehash: bool = False


class VirtualNameInput(Input):
    name: str = Field(min_length=1, max_length=80)


class VirtualRenameInput(VirtualNameInput):
    revision: int = Field(ge=1)


class VirtualRevisionInput(Input):
    revision: int = Field(ge=1)


class VirtualMembersInput(VirtualRevisionInput):
    image_ids: list[str] = Field(min_length=1, max_length=500)


class PrepareTrainingInput(Input):
    review: dict


class TrainInput(Input):
    dataset_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class ActivationInput(Input):
    model_version: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    acknowledge_experimental: bool = False
    rollback: bool = False


class JudgmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["relevant", "irrelevant"]
    expected_event_id: int | None = Field(default=None, ge=0)


class DuplicateReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: str = "personal"
    state: Literal["confirmed", "dismissed", "unreviewed"]


class WorkspacePreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: list[str] = Field(min_length=1, max_length=16)
    destination: str = Field(min_length=1, max_length=4096)


class WorkspaceImportInput(WorkspacePreviewInput):
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def create_app(
    *,
    index_dir: Path,
    models: Path = Path("models"),
    allowed_roots: tuple[Path, ...] | None = None,
    threads: int = 2,
    port: int = 8765,
    dev_origin: str | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    if not 1 <= threads <= 8 or not 1 <= port <= 65535:
        raise ValueError("Use 1–8 threads and a valid port")
    root = index_dir.resolve()
    from realopen_lens.workspace.identity import build, describe

    running_build = build()
    roots = tuple(
        p.expanduser().resolve()
        for p in ((Path.home(),) if allowed_roots is None else allowed_roots)
    )
    origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    if dev_origin is not None:
        from urllib.parse import urlparse

        parsed = urlparse(dev_origin)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1"}
            or parsed.path
        ):
            raise ValueError("Development origin must be an HTTP loopback origin")
        origins.add(dev_origin)
    token = secrets.token_urlsafe(32)
    cookie_name = f"lens_session_{port}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with writer_lock(root / "api-server"):
            manager = TaskManager(root, models.resolve(), threads, owns_server=True)
            app.state.manager = manager
            try:
                yield
            finally:
                await asyncio.to_thread(manager.close)

    app = FastAPI(
        title="RealOpen Lens", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def local_session(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return Response("Browser origin is not allowed", status_code=403)
        if request.url.path.startswith("/api/") and request.url.path != "/api/session":
            if not hmac.compare_digest(
                request.cookies.get(cookie_name, "").encode("utf-8"), token.encode("ascii")
            ):
                return Response("Open the local gallery to establish a session", status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; connect-src 'self'; "
            "style-src 'self'; script-src 'self'; frame-ancestors 'none'; object-src 'none'"
        )
        return response

    @app.exception_handler(ValueError)
    async def invalid(request: Request, error: ValueError):
        return Response(
            json.dumps({"detail": str(error)}), status_code=400, media_type="application/json"
        )

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(OSError)
    async def unavailable(request: Request, error: Exception):
        return Response(
            json.dumps({"detail": str(error)}), status_code=400, media_type="application/json"
        )

    @app.get("/api/session")
    def session():
        response = Response(
            json.dumps({"allowed_roots": [str(p) for p in roots]}), media_type="application/json"
        )
        response.set_cookie(cookie_name, token, httponly=True, samesite="strict")
        return response

    @app.get("/api/workspace")
    def workspace():
        return describe(root, models, running_build=running_build)

    @app.post("/api/tasks/workspace-preview", status_code=202)
    def preview_workspace(body: WorkspacePreviewInput):
        return app.state.manager.submit(
            "workspace_preview", {**body.model_dump(), "collection": "workspace"}
        )

    @app.post("/api/tasks/workspace-import", status_code=202)
    def import_workspace(body: WorkspaceImportInput):
        return app.state.manager.submit(
            "workspace_import", {**body.model_dump(), "collection": "workspace"}
        )

    @app.get("/api/collections")
    def collection_list():
        return reading.collections(root)

    @app.get("/api/collections/{collection:path}/images")
    def browse(collection: str, offset: int = 0, limit: int = 60):
        if not 0 <= offset <= 1_000_000 or not 1 <= limit <= 100:
            raise ValueError("Invalid page range")
        return reading.browse(root, collection, offset, limit)

    @app.get("/api/collections/{collection:path}/status")
    def status(collection: str):
        return collection_status(collection, root)

    @app.get("/api/collections/{collection:path}/images/{image_id}/{kind}")
    def image(
        collection: str,
        image_id: str,
        kind: Literal["thumbnail", "original"],
        generation: str | None = None,
    ):
        import re

        if not re.fullmatch(r"[a-f0-9]{64}", image_id):
            raise HTTPException(404, "Image not found")
        try:
            data, mime = reading.media(root, collection, image_id, kind == "thumbnail", generation)
        except (ValueError, OSError) as error:
            raise HTTPException(404, str(error)) from error
        return Response(data, media_type=mime)

    def submit(kind: str, body: dict):
        try:
            return app.state.manager.submit(kind, body)
        except ValueError as error:
            raise HTTPException(429, str(error)) from error

    from realopen_lens.organization import collections as organizing
    from realopen_lens.storage.organization import CollectionConflict

    @app.exception_handler(CollectionConflict)
    async def stale_virtual_collection(request: Request, error: CollectionConflict):
        return Response(
            json.dumps({"detail": str(error)}), status_code=409, media_type="application/json"
        )

    @app.post("/api/tasks/duplicates", status_code=202)
    def scan_duplicates(body: Input):
        return submit("duplicates", body.model_dump())

    @app.get("/api/duplicates")
    def duplicate_groups(collection: str = "personal", offset: int = 0, limit: int = 20):
        from realopen_lens.organization.duplicates import browse

        return browse(root, collection, offset, limit)

    @app.post("/api/duplicates/{pair_id}/review")
    def review_duplicate(pair_id: str, body: DuplicateReviewInput):
        from realopen_lens.organization.duplicates import review

        return review(root, body.collection, pair_id, body.state)

    @app.get("/api/virtual-collections")
    def virtual_collections(collection: str = "personal"):
        return organizing.list_collections(root, collection)

    @app.post("/api/virtual-collections", status_code=201)
    def create_virtual(body: VirtualNameInput):
        return organizing.create(root, body.collection, body.name)

    @app.patch("/api/virtual-collections/{virtual_id}")
    def rename_virtual(virtual_id: str, body: VirtualRenameInput):
        return organizing.rename(root, body.collection, virtual_id, body.name, body.revision)

    @app.delete("/api/virtual-collections/{virtual_id}")
    def delete_virtual(virtual_id: str, body: VirtualRevisionInput):
        organizing.remove(root, body.collection, virtual_id, body.revision)
        return {"deleted": True}

    @app.post("/api/virtual-collections/{virtual_id}/members")
    def add_virtual_members(virtual_id: str, body: VirtualMembersInput):
        return organizing.change_members(
            root, body.collection, virtual_id, body.image_ids, body.revision
        )

    @app.delete("/api/virtual-collections/{virtual_id}/members")
    def remove_virtual_members(virtual_id: str, body: VirtualMembersInput):
        return organizing.change_members(
            root, body.collection, virtual_id, body.image_ids, body.revision, remove=True
        )

    @app.get("/api/virtual-collections/{virtual_id}/images")
    def virtual_images(
        virtual_id: str,
        collection: str = "personal",
        offset: int = 0,
        limit: int = 60,
        revision: int | None = None,
        generation: str | None = None,
    ):
        return organizing.browse(root, collection, virtual_id, offset, limit, revision, generation)

    @app.post("/api/tasks/search", status_code=202)
    def search(body: SearchInput):
        return submit("search", body.model_dump())

    @app.post("/api/tasks/similar", status_code=202)
    def similar(body: SimilarInput):
        return submit("similar", body.model_dump())

    @app.post("/api/tasks/index", status_code=202)
    def index(body: IndexInput):
        path = Path(body.root).expanduser().resolve(strict=True)
        registered = [Path(c["root"]).resolve() for c in reading.collections(root)]
        if not path.is_dir() or not any(path.is_relative_to(p) for p in (*roots, *registered)):
            raise HTTPException(403, "Folder is outside the configured import roots")
        return submit("index", {**body.model_dump(), "root": str(path)})

    @app.post("/api/tasks/upload", status_code=202)
    async def upload(
        request: Request,
        collection: str = "personal",
        save_run: bool = False,
        virtual_collection: str | None = None,
        virtual_revision: int | None = None,
    ):
        ScopedInput(
            collection=collection,
            virtual_collection=virtual_collection,
            virtual_revision=virtual_revision,
        )
        directory = root / "api-uploads"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (identifier() + ".image")
        size = 0
        try:
            with path.open("wb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 64 * 1024 * 1024:
                        raise HTTPException(413, "Image exceeds 64 MiB")
                    stream.write(chunk)
            if not size:
                raise ValueError("Select a nonempty image")
            return submit(
                "similar",
                {
                    "collection": collection,
                    "save_run": save_run,
                    "limit": 50,
                    "upload": str(path),
                    "virtual_collection": virtual_collection,
                    "virtual_revision": virtual_revision,
                },
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @app.post("/api/training/snapshot")
    def feedback_snapshot():
        from realopen_lens.learning.dataset import export_snapshot
        from realopen_lens.models.manager import digest

        directory = root / "training-inputs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (identifier() + ".jsonl")
        count = export_snapshot(root, path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        template: dict = {
            "version": 1,
            "reviewer": "",
            "feedback_file": path.name,
            "feedback_sha256": digest(path),
            "independent_holdout": False,
            "runs": {r["run_id"]: {"family": "", "split": "", "query": r["query"]} for r in rows},
            "image_groups": {r["image_id"]: "" for r in rows},
        }
        from realopen_lens.learning.review import audit

        template["audit"] = {
            name: audit(root, name) for name in sorted({row["collection"] for row in rows})
        }
        for run_id, assignment in template["runs"].items():
            saved = feedback.saved_search(run_id, index_dir=root)
            selected_ids = {r["image_id"] for r in rows if r["run_id"] == run_id}
            assignment["results"] = [
                {
                    "image_id": r["image_id"],
                    "paths": r["paths"],
                    "label": next(
                        row["judgment"]
                        for row in rows
                        if row["run_id"] == run_id and row["image_id"] == r["image_id"]
                    ),
                }
                for r in saved["results"]
                if r["image_id"] in selected_ids
            ]
        return {"pairs": count, "review_template": template}

    @app.post("/api/tasks/prepare-training", status_code=202)
    def prepare_training(body: PrepareTrainingInput):
        import re
        from realopen_lens.learning.dataset import canonical
        from realopen_lens.storage.artifacts import atomic_bytes

        filename = body.review.get("feedback_file", "")
        if not isinstance(filename, str) or not re.fullmatch(r"[a-f0-9]{32}\.jsonl", filename):
            raise ValueError("Review must identify a local feedback snapshot")
        dataset_id = identifier()
        path = root / "training-inputs" / (dataset_id + ".review.json")
        atomic_bytes(path, canonical(body.review))
        return submit(
            "prepare_training",
            {
                "collection": body.collection,
                "review": str(path),
                "output": str(root / "training-datasets" / dataset_id),
            },
        )

    @app.post("/api/tasks/train", status_code=202)
    def train_model(body: TrainInput):
        path = root / "training-datasets" / body.dataset_id / "dataset.json"
        if not path.is_file():
            raise ValueError("Reviewed dataset does not exist")
        return submit("train", {"collection": body.collection, "dataset": str(path)})

    @app.get("/api/training/models")
    def trained_models():
        from realopen_lens.learning.registry import state, versions

        datasets = []
        for path in sorted((root / "training-datasets").glob("*/dataset.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            datasets.append({"id": path.parent.name, "queries": len(data["sessions"])})
        return {"versions": versions(root), "datasets": datasets, **state(root)}

    @app.post("/api/training/activation")
    def activate_model(body: ActivationInput):
        from realopen_lens.learning.registry import change

        return change(
            root,
            body.collection,
            body.model_version,
            acknowledge_experimental=body.acknowledge_experimental,
            rollback=body.rollback,
        )

    @app.get("/api/tasks")
    def tasks():
        return app.state.manager.list()

    @app.get("/api/tasks/{task_id}")
    def task(task_id: str):
        try:
            return app.state.manager.snapshot(task_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel(task_id: str):
        app.state.manager.cancel(task_id)
        return task(task_id)

    @app.get("/api/tasks/{task_id}/events")
    async def events(task_id: str, request: Request):
        task(task_id)

        async def stream():
            while not await request.is_disconnected():
                state = app.state.manager.snapshot(task_id)
                yield f"id: {state['revision']}\nevent: progress\ndata: {json.dumps(state)}\n\n"
                if state["status"] in TERMINAL:
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    from realopen_lens.storage.feedback import FeedbackConflict

    @app.exception_handler(FeedbackConflict)
    async def stale_feedback(request: Request, error: FeedbackConflict):
        return Response(
            json.dumps({"detail": str(error)}), status_code=409, media_type="application/json"
        )

    @app.get("/api/feedback/hidden")
    def hidden_feedback(
        query: str,
        collection: str = "personal",
        virtual_collection: str | None = None,
        virtual_revision: int | None = None,
        offset: int = 0,
        limit: int = 50,
    ):
        from realopen_lens.retrieval.memory import hidden

        return hidden(
            root,
            query,
            collection=collection,
            virtual_collection=virtual_collection,
            virtual_revision=virtual_revision,
            offset=offset,
            limit=limit,
        )

    @app.get("/api/feedback/runs/{run_id}")
    def saved(run_id: str):
        return feedback.saved_search(run_id, index_dir=root)

    @app.put("/api/feedback/runs/{run_id}/{image_id}")
    def judge(run_id: str, image_id: str, body: JudgmentInput):
        result = feedback.change_judgment(
            run_id, image_id, body.label, index_dir=root, expected_event_id=body.expected_event_id
        )
        return {**(result["judgment"] or {}), "memory": result["memory"]}

    @app.delete("/api/feedback/runs/{run_id}/{image_id}")
    def remove(run_id: str, image_id: str, expected_event_id: int | None = None):
        result = feedback.change_judgment(
            run_id, image_id, None, index_dir=root, expected_event_id=expected_event_id
        )
        return {"removed": True, "memory": result["memory"]}

    static = static_dir or Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/")
        def gallery():
            return FileResponse(static / "index.html")

    else:

        @app.get("/")
        def missing_gallery():
            return Response(
                "Build the gallery first: cd frontend && npm ci && npm run build", status_code=503
            )

    return app
