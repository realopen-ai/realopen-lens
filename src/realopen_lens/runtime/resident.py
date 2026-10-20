"""One owned, serialized query worker with bounded IPC and explicit lifetime."""

import atexit
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
from time import monotonic

import numpy as np

from realopen_lens.models.manager import artifact_path, artifacts, verify
from realopen_lens.runtime.jobs import environment, stop_worker


def _responses(stream, responses):
    try:
        while line := stream.readline(65537):
            if len(line) > 65536 or not line.endswith("\n"):
                raise ValueError("Query worker response exceeded its protocol limit")
            responses.put_nowait(json.loads(line))
        responses.put_nowait({"error": "Query worker exited before returning a response"})
    except Exception as error:
        try:
            responses.put_nowait({"error": f"Invalid query worker response: {error}"})
        except queue.Full:
            pass


def _send(stream, request: dict, responses):
    try:
        stream.write(json.dumps(request) + "\n")
        stream.flush()
    except Exception as error:
        try:
            responses.put_nowait({"error": f"Query worker write failed: {error}"})
        except queue.Full:
            pass


class ResidentQueryEncoder:
    """At most one subprocess, two lazy text models and one outstanding request.

    The first call binds model directory, precision and thread count. Model bytes
    are SHA-256 verified initially; changed filesystem signatures force a restart
    and complete verification. No query/result cache or network service is used.
    Failed requests are never retried silently. The next call can start a worker.
    """

    def __init__(self, *, timeout: float = 120):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Query timeout must be positive and finite")
        self.timeout = timeout
        self._lock = threading.Lock()
        self._closed = False
        self._identity: tuple | None = None
        self._signatures: tuple | None = None
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        atexit.register(self.close)

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _files(self, models: Path, precision: str) -> tuple:
        paths = [artifact_path(models, name, key) for name, key, _ in artifacts("int8", precision)]
        return tuple(
            (
                str(path.resolve()),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
            for path in paths
            for stat in [path.stat()]
        )

    def _verify(self, models: Path, threads: int, precision: str):
        if self._closed:
            raise RuntimeError("Resident query encoder is closed")
        if not 1 <= threads <= 8 or precision not in {"int8", "fp32"}:
            raise ValueError("Use 1–8 threads and int8 or fp32 precision")
        identity = (models.resolve(), threads, precision)
        if self._identity is not None and self._identity != identity:
            raise ValueError(
                "Resident session is bound to its model directory, threads and precision"
            )
        try:
            before = self._files(models, precision)
            if before != self._signatures:
                self._stop()
                self._signatures = None
                verify(models, "int8", precision)
                after = self._files(models, precision)
                if before != after:
                    raise ValueError("Model files changed during integrity verification")
                self._signatures = after
            self._identity = identity
        except BaseException:
            self._stop()
            raise

    def verify(self, models: Path, threads: int, precision: str):
        if not self._lock.acquire(timeout=self.timeout):
            raise TimeoutError("Resident query encoder is busy")
        try:
            self._verify(models, threads, precision)
        finally:
            self._lock.release()

    def _command(self) -> list[str]:
        return [sys.executable, "-m", "realopen_lens.runtime.query_server"]

    def _start(self, threads: int):
        self._queue = queue.Queue(maxsize=1)
        self._process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment(threads),
            start_new_session=sys.platform != "win32",
        )
        self._reader = threading.Thread(
            target=_responses,
            args=(self._process.stdout, self._queue),
            daemon=True,
            name="lens-query-responses",
        )
        self._reader.start()

    def encode(
        self, text: str, kind: str, models: Path, threads: int, precision: str
    ) -> np.ndarray:
        if kind not in {"siglip2", "e5"} or not text.strip() or len(text) > 4096:
            raise ValueError("Invalid query encoder request")
        deadline = monotonic() + self.timeout
        if not self._lock.acquire(timeout=self.timeout):
            raise TimeoutError("Resident query encoder is busy")
        try:
            self._verify(models, threads, precision)
            if self._process is not None and self._process.poll() is not None:
                self._stop()
            if self._process is None:
                self._start(threads)
            assert self._process is not None and self._process.stdin is not None
            request = {
                "query": text,
                "kind": kind,
                "models": str(models.resolve()),
                "threads": threads,
                "siglip_precision": precision,
            }
            self._writer = threading.Thread(
                target=_send,
                args=(self._process.stdin, request, self._queue),
                daemon=True,
                name="lens-query-request",
            )
            self._writer.start()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Query encoder exceeded its deadline")
            try:
                response = self._queue.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("Query encoder exceeded its deadline") from error
            if not isinstance(response, dict) or "vector" not in response:
                message = (
                    str(response.get("error", "invalid response"))[:1500]
                    if isinstance(response, dict)
                    else "Invalid response"
                )
                raise RuntimeError("Local query encoder failed: " + message)
            self._writer.join(timeout=5)
            self._writer = None
            vector = np.asarray(response["vector"], dtype=np.float32)
            if (
                vector.shape != (768 if kind == "siglip2" else 384,)
                or not np.isfinite(vector).all()
            ):
                raise ValueError("Invalid query embedding")
            return vector
        except BaseException:
            self._stop()
            raise
        finally:
            self._lock.release()

    def _stop(self):
        process, reader, writer = self._process, self._reader, self._writer
        self._process = self._reader = None
        self._writer = None
        if process is not None:
            if process.poll() is None:
                stop_worker(process.pid)
            process.wait(timeout=10)
            if reader is not None:
                reader.join(timeout=5)
            if writer is not None:
                writer.join(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    def close(self):
        with self._lock:
            self._closed = True
            self._stop()
        atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
