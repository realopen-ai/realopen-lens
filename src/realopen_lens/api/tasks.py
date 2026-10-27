"""Bounded application jobs, persisted operational state and reconnectable progress."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import threading
from time import time
from typing import Any

from realopen_lens.indexing.service import index_collection
from realopen_lens.learning.feedback import similar_and_record
from realopen_lens.retrieval.images import search_similar
from realopen_lens.retrieval.session import SearchSession
from realopen_lens.retrieval.library import LibrarySession
from realopen_lens.storage.artifacts import atomic_bytes
from realopen_lens.storage.catalog import identifier

TERMINAL = {"completed", "completed_with_errors", "failed", "cancelled", "interrupted"}


class Cancelled(KeyboardInterrupt):
    pass


class TaskManager:
    def __init__(
        self, index_dir: Path, models: Path, threads: int = 2, *, owns_server: bool = False
    ):
        self.owns_server = owns_server
        self.root, self.models, self.threads = index_dir, models, threads
        self.directory = index_dir / "api-tasks"
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.cancelled: set[str] = set()
        self.closed = False
        from realopen_lens.api.search_pages import SearchPages

        self.pages = SearchPages(index_dir)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lens-jobs")
        self.session: SearchSession | None = None
        self.identity: tuple | None = None
        self.library_session: LibrarySession | None = None
        uploads = index_dir / "api-uploads"
        if uploads.exists():
            for upload in uploads.glob("*.image"):
                upload.unlink(missing_ok=True)
        if self.directory.exists():
            paths = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
            for obsolete in paths[:-50]:
                obsolete.unlink(missing_ok=True)
            for path in paths[-50:]:
                state = json.loads(path.read_text(encoding="utf-8"))
                if state["status"] not in TERMINAL:
                    state.update(
                        status="interrupted",
                        phase="Server stopped; submit again",
                        finished_at=time(),
                        revision=state["revision"] + 1,
                    )
                state["result"] = None
                self.tasks[state["id"]] = state
                self._persist(state)

        from realopen_lens.api.refresh import FolderRefresh

        self.auto_refresh = FolderRefresh(self)

    def _persist(self, state: dict):
        payload = {k: v for k, v in state.items() if k != "result"}
        atomic_bytes(
            self.directory / (state["id"] + ".json"),
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    def snapshot(self, task_id: str) -> dict:
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError("Task not found")
            state = dict(self.tasks[task_id])
            state["elapsed_seconds"] = max(
                0, (state.get("finished_at") or time()) - state["created_at"]
            )
            return state

    def list(self, *, summary: bool = False) -> list[dict]:
        with self.lock:
            states = [self.snapshot(key) for key in reversed(self.tasks)]
            if summary:
                for state in states:
                    state["result"] = None
            return states

    def update(self, task_id: str, **values):
        with self.lock:
            state = self.tasks[task_id]
            state.update(values)
            state["revision"] += 1
            self._persist(state)

    def submit(self, kind: str, request: dict) -> dict:
        with self.lock:
            if self.closed:
                raise ValueError("Server is shutting down")
            if sum(s["status"] not in TERMINAL for s in self.tasks.values()) >= 4:
                raise ValueError("Job queue is full; wait for a task to finish")
            while len(self.tasks) >= 50:
                old = next(
                    (key for key, value in self.tasks.items() if value["status"] in TERMINAL), None
                )
                if old is None:
                    break
                self.tasks.pop(old)
                (self.directory / (old + ".json")).unlink(missing_ok=True)
            task_id = identifier()
            state = {
                "id": task_id,
                "kind": kind,
                "virtual_collection": request.get("virtual_collection"),
                "virtual_revision": request.get("virtual_revision"),
                "collection": request["collection"],
                "collection_id": request.get("collection_id"),
                "all_folders": request.get("all_folders", False),
                "origin": request.get("origin", "manual"),
                "status": "queued",
                "phase": "queued",
                "created_at": time(),
                "revision": 0,
                "result": None,
                "error": None,
                "progress": {},
            }
            self.tasks[task_id] = state
            self._persist(state)
            self.executor.submit(self._run, task_id, kind, request)
            return self.snapshot(task_id)

    def cancel(self, task_id: str):
        with self.lock:
            state = self.snapshot(task_id)
            if state["status"] not in TERMINAL:
                self.cancelled.add(task_id)
                self.auto_refresh.cancelled(state)
                self.update(task_id, phase="cancellation_requested")

    def manage_folder(self, name: str, collection_id: str, new_name: str | None) -> dict:
        from realopen_lens.organization.folders import change
        from realopen_lens.storage.organization import CollectionConflict

        with self.lock:
            if self.closed or any(s["status"] not in TERMINAL for s in self.tasks.values()):
                raise CollectionConflict("Wait for active tasks to finish before changing folders")
            self._release_session()
            result = change(self.root, name, collection_id, new_name)
            if new_name is None:
                self.auto_refresh.forget(collection_id)
            return result

    def refresh_folder(
        self,
        name: str,
        collection_id: str,
        *,
        automatic: bool = False,
        languages: str | None = None,
        source_identity: Sequence[int] | None = None,
    ) -> dict:
        from realopen_lens.organization.folders import refresh_options
        from realopen_lens.storage.organization import CollectionConflict

        with self.lock:
            if any(
                s["status"] not in TERMINAL and s["collection"] == name for s in self.tasks.values()
            ):
                raise CollectionConflict(
                    "This folder already has an active task; wait for it to finish"
                )
            request = refresh_options(self.root, name, collection_id)
            if languages is not None:
                from realopen_lens.ocr.preferences import validate

                request["languages"] = validate(languages)
            request.update(
                collection_id=collection_id, origin="automatic" if automatic else "manual"
            )
            if automatic:
                request["source_identity"] = source_identity
            return self.submit("index", request)

    def _retrieve(self, kind: str, request: dict, progress, options: dict) -> dict:
        collection = request["collection"]
        if request.get("all_folders"):
            if self.library_session is None:
                self._release_session()
                self.library_session = LibrarySession(
                    index_dir=self.root, models=self.models, threads=self.threads
                )
            if kind == "similar":
                payload = self.library_session.similar(
                    Path(request["upload"]) if request.get("upload") else None,
                    image_id=request.get("image_id"),
                    collection=collection,
                    limit=request["limit"],
                    deep_search=request.get("deep_search", False),
                    save_run=request.get("save_run", False),
                    defer_record=request.get("defer_record", False),
                    on_progress=progress,
                    virtual_collection=request.get("virtual_collection"),
                    virtual_revision=request.get("virtual_revision"),
                )
            else:
                payload = self.library_session.search(
                    request["query"],
                    limit=request["limit"],
                    deep_search=request.get("deep_search", False),
                    mode=request["mode"],
                    use_feedback=request.get("use_feedback", True),
                    interpret_metadata=request.get("interpret_metadata", True),
                    save_run=request.get("save_run", False),
                    defer_record=request.get("defer_record", False),
                    on_progress=progress,
                    virtual_collection=request.get("virtual_collection"),
                    virtual_revision=request.get("virtual_revision"),
                )
        elif kind == "search":
            from realopen_lens.indexing.service import collection_status

            precision = collection_status(collection, self.root)["siglip_precision"]
            identity = (collection, precision)
            if self.identity != identity:
                self._release_session()
                self.session = SearchSession(
                    collection=collection,
                    index_dir=self.root,
                    models=self.models,
                    threads=self.threads,
                )
                self.identity = identity
            assert self.session is not None
            feedback_state: dict = {}
            interpretation: dict = {}
            kwargs = {
                "interpret_metadata": request.get("interpret_metadata", True),
                "metadata_trace": interpretation,
                "use_feedback": request.get("use_feedback", True),
                "feedback_trace": feedback_state,
                "limit": request["limit"],
                "mode": request["mode"],
                "deep_search": request.get("deep_search", False),
                "on_progress": progress,
                "virtual_collection": request.get("virtual_collection"),
                "virtual_revision": request.get("virtual_revision"),
            }
            ranking: dict = {"kind": request["mode"], "sha256": None}
            if request.get("save_run"):
                recorded = self.session.search_recorded(
                    request["query"], defer_record=request.get("defer_record", False), **kwargs
                )
                results, run_id = recorded.results, recorded.run_id
                from realopen_lens.learning.feedback import saved_search

                ranking["sha256"] = (recorded.payload or saved_search(run_id, index_dir=self.root))[
                    "ranking"
                ]["reranker_sha256"]
            else:
                trace: dict = {}
                if request["mode"] == "hybrid":
                    kwargs["trace"] = trace
                results = self.session.search(request["query"], **kwargs)
                run_id = None
                ranking["sha256"] = trace.get("reranker_sha256")
            if ranking["sha256"]:
                ranking["kind"] = "logistic"
            elif interpretation and not interpretation["visual_query"]:
                ranking["kind"] = "metadata"
            payload = {
                "_record_payload": (
                    recorded.payload
                    if request.get("defer_record") and request.get("save_run")
                    else None
                ),
                "results": [asdict(r) for r in results],
                "run_id": run_id,
                "ranking": ranking,
                "feedback": feedback_state,
                "interpretation": interpretation or None,
            }
        else:
            self._release_session()
            image_options = {
                "image_id": request.get("image_id"),
                "collection": collection,
                "limit": request["limit"],
                "deep_search": request.get("deep_search", False),
                "virtual_collection": request.get("virtual_collection"),
                "virtual_revision": request.get("virtual_revision"),
                **options,
            }
            path = Path(request["upload"]) if request.get("upload") else None
            if request.get("save_run"):
                recorded = similar_and_record(
                    path, defer_record=request.get("defer_record", False), **image_options
                )
                results, run_id = recorded.results, recorded.run_id
            else:
                results = search_similar(path, **image_options)
                run_id = None
            payload = {
                "results": [asdict(r) for r in results],
                "run_id": run_id,
                "_record_payload": (
                    recorded.payload
                    if request.get("defer_record") and request.get("save_run")
                    else None
                ),
            }
        return payload

    def _release_session(self):
        if self.library_session is not None:
            self.library_session.close()
            self.library_session = None
        if self.session is not None:
            self.session.close()
        self.session, self.identity = None, None

    def _run(self, task_id: str, kind: str, request: dict):
        def progress(values: dict):
            with self.lock:
                if task_id in self.cancelled:
                    raise Cancelled()
            # Keep operational journals small; full indexing failures remain in the catalog.
            small = {
                k: v
                for k, v in values.items()
                if k not in {"failures", "degradations", "current_paths"}
            }
            self.update(task_id, phase=values.get("phase") or "scanning", progress=small)

        try:
            progress({"phase": "starting"})
            self.update(task_id, status="running")
            options: dict[str, Any] = {
                "index_dir": self.root,
                "models": self.models,
                "threads": self.threads,
                "on_progress": progress,
            }
            collection = request["collection"]
            if kind in {"search", "similar"} and request.get("virtual_collection"):
                from realopen_lens.organization.collections import parent, resolve_scope

                found = (
                    {"id": "library-v1"}
                    if request.get("all_folders")
                    else parent(self.root, collection)
                )
                _, scope = resolve_scope(
                    self.root,
                    found["id"],
                    request["virtual_collection"],
                    request.get("virtual_revision"),
                )
                request = {**request, "virtual_revision": scope["revision"]}
                self.update(task_id, virtual_revision=scope["revision"])
            if kind == "setup":
                from realopen_lens.runtime.readiness import inspect

                payload = inspect(
                    self.root,
                    self.models,
                    siglip_precision=request["siglip_precision"],
                    languages=request["languages"],
                    source=Path(request["source"]) if request.get("source") else None,
                    on_progress=progress,
                )
                status = "completed"  # Readiness is reported explicitly; execution isn't readiness.
            elif kind in {"workspace_preview", "workspace_import"}:
                from realopen_lens.workspace.service import preview, initialize

                self._release_session()
                sources = [Path(value) for value in request["sources"]]
                destination = Path(request["destination"])
                if kind == "workspace_preview":
                    payload = preview(sources, destination, progress)
                else:
                    payload = initialize(
                        sources,
                        destination,
                        request["plan_sha256"],
                        models=self.models,
                        on_progress=progress,
                        owned_server=self.root if self.owns_server else None,
                    )
                status = "completed"
            elif kind in {"prepare_training", "train"}:
                self._release_session()
                if kind == "prepare_training":
                    from realopen_lens.learning.dataset import prepare

                    payload = prepare(
                        self.root, Path(request["review"]), Path(request["output"]), progress
                    )
                else:
                    from realopen_lens.learning.workflow import train

                    payload = train(Path(request["dataset"]), self.root, self.models, progress)
                status = payload.get("status", "completed")
            elif kind == "duplicates":
                from realopen_lens.organization.duplicates import scan

                self._release_session()
                payload = scan(self.root, collection, progress)
                status = payload["status"]
            elif kind == "index":
                self._release_session()
                if request.get("origin") == "automatic":
                    options.update(
                        expected_source_identity=request["source_identity"],
                        expected_collection_id=request["collection_id"],
                    )
                result = index_collection(
                    Path(request["root"]),
                    collection=collection,
                    languages=request["languages"],
                    siglip_precision=request["siglip_precision"],
                    rehash=request.get("rehash", False),
                    **options,
                )
                payload = asdict(result)
                status = result.status
            elif kind in {"search", "similar"}:
                if request.get("paginate"):
                    payload = self.pages.create(
                        {**request, "kind": kind},
                        lambda values: self._retrieve(kind, values, progress, options),
                    )
                else:
                    payload = self._retrieve(kind, request, progress, options)
                status = "completed"
            if kind in {"search", "similar"}:
                payload.pop("_record_payload", None)
                payload["virtual_collection"] = request.get("virtual_collection")
                payload["virtual_revision"] = request.get("virtual_revision")
            # Once an operation has committed (index/feedback), report its real outcome.
            self.update(
                task_id, status=status, phase="finished", result=payload, finished_at=time()
            )
        except Cancelled:
            self._release_session()
            self.update(task_id, status="cancelled", phase="cancelled", finished_at=time())
        except Exception as error:
            self._release_session()
            self.update(
                task_id,
                status="failed",
                phase="failed",
                error=str(error)[-2000:],
                finished_at=time(),
            )
        finally:
            if request.get("upload"):
                Path(request["upload"]).unlink(missing_ok=True)
            with self.lock:
                self.cancelled.discard(task_id)
                self.auto_refresh.finished(self.tasks[task_id])

    def close(self):
        with self.lock:
            self.closed = True
            self.cancelled.update(k for k, s in self.tasks.items() if s["status"] not in TERMINAL)
        self.auto_refresh.close()
        self.executor.shutdown(wait=True)
        self._release_session()
