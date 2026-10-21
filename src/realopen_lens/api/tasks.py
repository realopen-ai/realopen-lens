"""Bounded application jobs, persisted operational state and reconnectable progress."""

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
from realopen_lens.storage.artifacts import atomic_bytes
from realopen_lens.storage.catalog import identifier

TERMINAL = {"completed", "completed_with_errors", "failed", "cancelled", "interrupted"}


class Cancelled(KeyboardInterrupt):
    pass


class TaskManager:
    def __init__(self, index_dir: Path, models: Path, threads: int = 2):
        self.root, self.models, self.threads = index_dir, models, threads
        self.directory = index_dir / "api-tasks"
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.cancelled: set[str] = set()
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lens-jobs")
        self.session: SearchSession | None = None
        self.identity: tuple | None = None
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
                    )
                state["result"] = None
                self.tasks[state["id"]] = state
                self._persist(state)

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

    def list(self) -> list[dict]:
        with self.lock:
            return [self.snapshot(key) for key in reversed(self.tasks)]

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
                "collection": request["collection"],
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
                self.update(task_id, phase="cancellation_requested")

    def _release_session(self):
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
            if kind in {"prepare_training", "train"}:
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
            elif kind == "index":
                self._release_session()
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
                kwargs = {
                    "limit": request["limit"],
                    "mode": request["mode"],
                    "on_progress": progress,
                }
                ranking: dict = {"kind": request["mode"], "sha256": None}
                if request.get("save_run"):
                    recorded = self.session.search_recorded(request["query"], **kwargs)
                    results, run_id = recorded.results, recorded.run_id
                    from realopen_lens.learning.feedback import saved_search

                    ranking["sha256"] = saved_search(run_id, index_dir=self.root)["ranking"][
                        "reranker_sha256"
                    ]
                else:
                    trace: dict = {}
                    if request["mode"] == "hybrid":
                        kwargs["trace"] = trace
                    results = self.session.search(request["query"], **kwargs)
                    run_id = None
                    ranking["sha256"] = trace.get("reranker_sha256")
                if ranking["sha256"]:
                    ranking["kind"] = "logistic"
                payload = {
                    "results": [asdict(r) for r in results],
                    "run_id": run_id,
                    "ranking": ranking,
                }
                status = "completed"
            else:
                self._release_session()
                image_options = {
                    "image_id": request.get("image_id"),
                    "collection": collection,
                    "limit": request["limit"],
                    **options,
                }
                path = Path(request["upload"]) if request.get("upload") else None
                if request.get("save_run"):
                    recorded = similar_and_record(path, **image_options)
                    results, run_id = recorded.results, recorded.run_id
                else:
                    results = search_similar(path, **image_options)
                    run_id = None
                payload = {"results": [asdict(r) for r in results], "run_id": run_id}
                status = "completed"
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

    def close(self):
        with self.lock:
            self.closed = True
            self.cancelled.update(k for k, s in self.tasks.items() if s["status"] not in TERMINAL)
        self.executor.shutdown(wait=True)
        self._release_session()
