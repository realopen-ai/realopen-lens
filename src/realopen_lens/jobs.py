"""One index writer and bounded native workers, independent of the benchmark harness."""

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep

import psutil

from realopen_lens.catalog import Catalog
from realopen_lens.index_files import valid_analysis

logger = logging.getLogger(__name__)


def environment(threads: int) -> dict[str, str]:
    return {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
    }


@contextmanager
def writer_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "indexing.lock").open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt

            lock.seek(0)
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError("Another indexing job is running; use 'lens status'") from error
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError("Another indexing job is running; use 'lens status'") from error
        try:
            yield
        finally:
            if sys.platform == "win32":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def stop_worker(pid: int, created: float | None = None):
    """Terminate only an owned stage, including its OCR descendants."""
    if created is not None:
        try:
            if abs(psutil.Process(pid).create_time() - created) > 0.01:
                return  # PID has been reused by another process.
        except psutil.NoSuchProcess:
            return
    if sys.platform != "win32":
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
            psutil.wait_procs([parent, *children], timeout=5)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                check=True,
            )


def recover(catalog: Catalog):
    for row in catalog.db.execute(
        "SELECT id,worker_pid,worker_created FROM jobs WHERE status='running'"
    ).fetchall():
        logger.warning("Recovering interrupted job %s; completed stages will be reused", row["id"])
        if row["worker_pid"] and row["worker_created"] is not None:
            stop_worker(row["worker_pid"], row["worker_created"])
        catalog.finish(
            row["id"], "interrupted", error="Previous coordinator stopped; cached stages can resume"
        )


def supervise(
    catalog: Catalog,
    job_id: str,
    stage: str,
    command: list[str],
    *,
    threads: int,
    item_timeout: float = 90,
    startup_timeout: float = 120,
):
    config = json.loads(catalog.job(job_id)["config"])
    log = catalog.root / "logs" / (job_id + ".log")
    log.parent.mkdir(exist_ok=True)
    logger.info("[%s] Worker diagnostics: %s", stage, log)
    while True:
        catalog.heartbeat(job_id, stage, None, None)
        offset = log.stat().st_size if log.exists() else 0
        with log.open("ab") as stream, log.open("r", encoding="utf-8", errors="replace") as reader:
            reader.seek(offset)

            def forward_logs():
                if config.get("log", False):
                    content = reader.read()
                    if content:
                        sys.stderr.write(content)
                        sys.stderr.flush()

            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=stream,
                env=environment(threads),
                start_new_session=sys.platform != "win32",
            )
            try:
                created = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                created = None
            catalog.worker(job_id, process.pid, created)
            started = monotonic()
            failure = None
            try:
                while process.poll() is None:
                    forward_logs()
                    row = catalog.job(job_id)
                    if row["current_image"] and monotonic() - row["item_started"] > item_timeout:
                        failure = "Native stage exceeded the per-image deadline"
                        stop_worker(process.pid)
                        break
                    if row["current_image"] is None and monotonic() - started > startup_timeout:
                        raise TimeoutError(f"{stage} worker startup exceeded its deadline")
                    sleep(0.1)
                process.wait(timeout=10)
                forward_logs()
                if process.returncode == 0:
                    return
                stop_worker(process.pid)  # Reap any descendants after a native crash.
                row = catalog.job(job_id)
                image_id = row["current_image"]
                if image_id is None:
                    raise RuntimeError(f"{stage} worker failed before processing; inspect {log}")
                existing = catalog.analysis(config["pipeline"], image_id, stage)
                if not valid_analysis(catalog, existing):
                    message = (
                        failure or f"Native {stage} worker exited with code {process.returncode}"
                    )
                    catalog.save_analysis(
                        config["pipeline"],
                        image_id,
                        stage,
                        {
                            "status": "error",
                            "error": message,
                            "error_type": "NativeTimeout" if failure else "NativeWorkerExit",
                        },
                        job_id=job_id,
                    )
                    catalog.fail_stage(job_id, image_id, stage, message)
                    paths = [r["path"] for r in catalog.files(job_id) if r["image_id"] == image_id]
                    logger.warning("[%s] Failed paths %r: %s", stage, paths, message)
                logger.warning(
                    "[%s] Restarting worker for remaining images; inspect %s", stage, log
                )
                # Restart: completed records stay durable and failed files are excluded.
            finally:
                if process.poll() is None:
                    stop_worker(process.pid)
                    process.wait(timeout=10)
                forward_logs()
                catalog.worker(job_id, None, None)
