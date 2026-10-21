"""Live stderr streaming preserves the indexing result protocol."""

import json
import subprocess
import sys
import textwrap


def test_log_streams_before_completion_and_keeps_result_protocol_clean(
    tmp_path, monkeypatch, capfd
):
    from concurrent.futures import ThreadPoolExecutor
    from time import monotonic, sleep
    from realopen_lens.indexing import service as indexing

    original_popen = subprocess.Popen
    marker, gate = tmp_path / "started", tmp_path / "finish"
    options = []
    script = textwrap.dedent("""
        import json, sys, time
        from pathlib import Path
        options = json.loads(Path(sys.argv[1]).read_text())
        print('live progress before completion', file=sys.stderr, flush=True)
        Path(sys.argv[2]).write_text('started')
        while not Path(sys.argv[3]).exists():
            time.sleep(0.01)
        print(json.dumps(dict(job_id='test', status='completed', collection='personal',
                             files=0, images=0, reused_images=0, failed_files=0,
                             removed_files=0, generation='test', seconds=0.1)))
        """)

    def worker(command, **kwargs):
        from pathlib import Path

        options.append(json.loads(Path(command[-1]).read_text()))
        return original_popen(
            [sys.executable, "-c", script, command[-1], str(marker), str(gate)], **kwargs
        )

    monkeypatch.setattr(indexing.subprocess, "Popen", worker)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            indexing.index_collection, tmp_path, index_dir=tmp_path / "index", log=True
        )
        try:
            deadline = monotonic() + 15
            while not marker.exists() and not future.done() and monotonic() < deadline:
                sleep(0.01)
            assert marker.exists() and not future.done()
            assert "live progress before completion" in capfd.readouterr().err
        finally:
            gate.touch()
        assert future.result(timeout=15).status == "completed"
    indexing.index_collection(tmp_path, index_dir=tmp_path / "index")
    assert capfd.readouterr().err == ""
    assert [item["log"] for item in options] == [True, False]


def test_progress_observer_runs_before_child_completion_and_reaps_on_error(tmp_path, monkeypatch):
    import pytest
    from realopen_lens.indexing import service as indexing

    original_popen = subprocess.Popen
    processes = []

    def worker(command, **kwargs):
        process = original_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(indexing.subprocess, "Popen", worker)
    observed = []

    def progress(state):
        assert processes[0].poll() is None
        observed.append(state)
        raise RuntimeError("observer stopped")

    with pytest.raises(RuntimeError, match="observer stopped"):
        indexing.index_collection(tmp_path, index_dir=tmp_path / "index", on_progress=progress)
    assert observed and processes[0].poll() is not None
    assert not list((tmp_path / "index" / "requests").glob("*.json"))
