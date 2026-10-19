"""Supervise stage workers and isolate native hangs/crashes to their active item."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from time import monotonic, sleep

import psutil


def stop_tree(process: subprocess.Popen) -> None:
    """Kill only this job's child tree, then reap its direct child."""
    if sys.platform != "win32":
        # run_stage creates a new session, so this group contains only this job.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        return
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    except (psutil.AccessDenied, OSError):
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            timeout=10,
            check=True,
        )
        process.wait(timeout=10)
        return
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.kill()
    process.wait(timeout=10)
    psutil.wait_procs(children, timeout=5)


def run_stage(
    command: list[str],
    db: sqlite3.Connection,
    stage: str,
    env: dict[str, str],
    log: Path,
    record_failure: Callable[[str, dict], None],
    image_timeout: float = 90,
    startup_timeout: float = 120,
    stage_timeout: float = 3600,
    poll_interval: float = 0.2,
) -> list[dict]:
    events: list[dict] = []
    begin = monotonic()
    while True:
        with db:
            db.execute("DELETE FROM metadata WHERE key=?", (stage + "_current",))
        with log.open("ab") as stream:
            process = subprocess.Popen(
                command, env=env, stdout=stream, stderr=stream, start_new_session=os.name != "nt"
            )
            launched = monotonic()
            failure = None
            current = None
            try:
                while process.poll() is None:
                    now = monotonic()
                    if now - begin > stage_timeout:
                        raise TimeoutError(f"{stage} exceeded whole-stage budget")
                    row = db.execute(
                        "SELECT value FROM metadata WHERE key=?", (stage + "_current",)
                    ).fetchone()
                    current = json.loads(row[0]) if row else None
                    if current and now - current["started_at"] > image_timeout:
                        failure = {
                            "status": "timeout",
                            "error_type": "NativeStageTimeout",
                            "error": f"{stage} exceeded {image_timeout}s for this item",
                            "seconds": now - current["started_at"],
                            "warnings": [],
                            "peak_tree_rss_mib": None,
                        }
                        stop_tree(process)
                        break
                    if not current and now - launched > startup_timeout:
                        raise TimeoutError(f"{stage} worker startup exceeded {startup_timeout}s")
                    sleep(poll_interval)
                if process.returncode == 0:
                    return events
                stop_tree(process)
                # A native crash can occur between polls. Read the last committed heartbeat.
                row = db.execute(
                    "SELECT value FROM metadata WHERE key=?", (stage + "_current",)
                ).fetchone()
                if failure is None:
                    current = json.loads(row[0]) if row else current
                if current is None:
                    raise RuntimeError(f"{stage} failed before an item started; inspect {log}")
                existing = db.execute(
                    "SELECT 1 FROM stages WHERE image_id=? AND stage=?", (current["id"], stage)
                ).fetchone()
                if not existing:
                    failure = failure or {
                        "status": "error",
                        "error_type": "NativeWorkerExit",
                        "error": f"{stage} worker exited with code {process.returncode}",
                        "seconds": monotonic() - current["started_at"],
                        "warnings": [],
                        "peak_tree_rss_mib": None,
                    }
                    record_failure(current["id"], failure)
                    events.append({"stage": stage, "id": current["id"], **failure})
                else:
                    events.append(
                        {
                            "stage": stage,
                            "event": "worker exited after item committed",
                            "returncode": process.returncode,
                            "id": current["id"],
                        }
                    )
            except BaseException:
                if process.poll() is None:
                    stop_tree(process)
                raise
