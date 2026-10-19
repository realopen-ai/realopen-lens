"""Real subprocess tests for writer exclusion and native-stage failure isolation."""

import json
import subprocess
import sys
import textwrap

import pytest

from realopen_lens.storage.catalog import Catalog
from realopen_lens.runtime.jobs import recover, supervise, writer_lock


@pytest.mark.parametrize(
    "failure,corrupt_cache,stage",
    [
        ("timeout", False, "vision"),
        ("crash", False, "vision"),
        ("crash", True, "vision"),
        ("timeout", False, "ocr"),
        ("crash", False, "ocr"),
        ("crash", False, "semantic"),
    ],
)
def test_failed_native_item_does_not_block_later_items(tmp_path, failure, corrupt_cache, stage):
    catalog = Catalog(tmp_path)
    collection = catalog.ensure_collection("test", tmp_path / "images")
    job = catalog.create_job(collection["id"], {"pipeline": "test"})
    catalog.add_file(job, "bad.png", "bad", {})
    catalog.add_file(job, "good.png", "good", {})
    if corrupt_cache:
        catalog.save_analysis("test", "bad", "vision", {"status": "ok"}, "missing.npy", "invalid")
    script = textwrap.dedent("""
        import os, sys, time
        from pathlib import Path
        from realopen_lens.storage.catalog import Catalog
        from realopen_lens.storage.artifacts import attempted, valid_analysis
        catalog = Catalog(Path(sys.argv[1]))
        job, failure, stage = sys.argv[2:]
        for row in catalog.files(job):
            if row['status'] == 'error':
                continue
            cached = catalog.analysis('test', row['image_id'], stage)
            if valid_analysis(catalog, cached) or attempted(cached, job):
                continue
            catalog.heartbeat(job, stage, row['image_id'], time.monotonic())
            if row['image_id'] == 'bad':
                if failure == 'crash':
                    os._exit(17)
                time.sleep(30)
            catalog.save_analysis('test', row['image_id'], stage, {'status': 'ok'}, job_id=job)
        catalog.close()
        """)
    supervise(
        catalog,
        job,
        stage,
        [sys.executable, "-c", script, str(tmp_path), job, failure, stage],
        threads=1,
        item_timeout=0.6,
        startup_timeout=15,
    )
    assert catalog.files(job)[0]["status"] == ("error" if stage == "vision" else "degraded")
    bad = json.loads(catalog.analysis("test", "bad", stage)["payload"])
    assert bad["error_type"] == ("NativeTimeout" if failure == "timeout" else "NativeWorkerExit")
    assert catalog.analysis("test", "good", stage)["status"] == "ok"
    assert catalog.job(job)["worker_pid"] is None
    catalog.close()


def test_writer_lock_excludes_another_process_and_releases(tmp_path):
    script = (
        "from pathlib import Path; import sys; "
        "from realopen_lens.runtime.jobs import writer_lock\n"
        "with writer_lock(Path(sys.argv[1])): pass\n"
    )
    command = [sys.executable, "-c", script, str(tmp_path)]
    with writer_lock(tmp_path):
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert result.returncode != 0 and "Another indexing job" in result.stderr
    assert subprocess.run(command, capture_output=True, timeout=15).returncode == 0


def test_stale_recovery_never_kills_a_pid_without_creation_time(tmp_path, monkeypatch):
    from realopen_lens.runtime import jobs

    catalog = Catalog(tmp_path)
    collection = catalog.ensure_collection("test", tmp_path / "images")
    job = catalog.create_job(collection["id"], {})
    catalog.worker(job, 123456, None)

    def unexpected(*args):
        pytest.fail("Cannot identify a stale PID without its original creation time")

    monkeypatch.setattr(jobs, "stop_worker", unexpected)
    recover(catalog)
    assert catalog.job(job)["status"] == "interrupted"
    catalog.close()


def test_worker_startup_failure_is_fatal_not_a_success(tmp_path):
    catalog = Catalog(tmp_path)
    collection = catalog.ensure_collection("test", tmp_path / "images")
    job = catalog.create_job(collection["id"], {"pipeline": "test"})
    with pytest.raises(RuntimeError, match="failed before processing"):
        supervise(
            catalog,
            job,
            "vision",
            [sys.executable, "-c", "raise RuntimeError('startup')"],
            threads=1,
        )
    assert catalog.job(job)["worker_pid"] is None
    catalog.close()
