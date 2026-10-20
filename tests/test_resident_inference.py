"""Resident worker lifetime, fault isolation, model identity and bounded reuse."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import threading

import numpy as np
import psutil
import pytest

from realopen_lens.runtime import resident


@pytest.fixture
def encoder(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    model.write_bytes(b"pinned")
    checks = []
    monkeypatch.setattr(resident, "artifacts", lambda *a: [("model", "text", {})])
    monkeypatch.setattr(resident, "artifact_path", lambda *a: model)

    def verify(*args):
        checks.append(args)
        if model.read_bytes() != b"pinned":
            raise ValueError("Model integrity check failed")

    monkeypatch.setattr(resident, "verify", verify)
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json,os,sys,time
for line in sys.stdin:
 r=json.loads(line);q=r['query']
 if q=='crash':os._exit(17)
 if q=='hang':time.sleep(60)
 if q=='invalid':print('{"vector": [0]}',flush=True);continue
 if q=='oversize':print('x'*70000,flush=True);continue
 if q=='slow':time.sleep(.05)
 print(json.dumps({'vector':[len(q)]*(768 if r['kind']=='siglip2' else 384)}),flush=True)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        resident.ResidentQueryEncoder, "_command", lambda self: [sys.executable, "-u", str(worker)]
    )
    with resident.ResidentQueryEncoder(timeout=3) as client:
        yield client, tmp_path, model, checks


def encode(client, models, text="hello", kind="siglip2", precision="int8"):
    return client.encode(text, kind, models, 2, precision)


def test_one_worker_reuses_verified_files_and_shuts_down(encoder):
    client, root, _, checks = encoder
    assert encode(client, root).shape == (768,)
    pid = client.pid
    assert encode(client, root, kind="e5").shape == (384,)
    assert client.pid == pid and len(checks) == 1
    with pytest.raises(ValueError, match="bound"):
        encode(client, root, precision="fp32")
    assert client.pid is None and not psutil.pid_exists(pid)
    encode(client, root)
    pid = client.pid
    client.close()
    client.close()
    assert not psutil.pid_exists(pid)
    with pytest.raises(RuntimeError, match="closed"):
        encode(client, root)


@pytest.mark.parametrize(
    "query,exception",
    [
        ("hang", TimeoutError),
        ("crash", RuntimeError),
        ("invalid", ValueError),
        ("oversize", RuntimeError),
    ],
)
def test_failed_request_reaps_worker_and_next_request_recovers(encoder, query, exception):
    client, root, _, _ = encoder
    encode(client, root)
    pid = client.pid
    client.timeout = 0.3
    with pytest.raises(exception):
        encode(client, root, query)
    assert client.pid is None and not psutil.pid_exists(pid)
    client.timeout = 3
    np.testing.assert_array_equal(encode(client, root), np.full(768, 5, dtype=np.float32))


def test_mutated_model_stops_worker_before_reuse(encoder):
    client, root, model, checks = encoder
    encode(client, root)
    pid = client.pid
    model.write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        encode(client, root)
    assert not psutil.pid_exists(pid) and len(checks) == 2
    model.write_bytes(b"pinned")
    encode(client, root)
    assert len(checks) == 3


def test_concurrent_requests_are_serialized_and_busy_wait_is_bounded(encoder):
    client, root, _, checks = encoder
    with ThreadPoolExecutor(max_workers=3) as pool:
        vectors = list(pool.map(lambda q: encode(client, root, q), ["one", "slow", "three"]))
    assert [v[0] for v in vectors] == [3, 4, 5] and len(checks) == 1
    pid = client.pid
    client.timeout = 0.05
    client._lock.acquire()
    try:
        with pytest.raises(TimeoutError, match="busy"):
            encode(client, root)
    finally:
        client._lock.release()
    assert client.pid == pid  # A waiting caller cannot kill another caller's request.


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        resident.ResidentQueryEncoder(timeout=timeout)


def test_server_lazily_loads_only_two_models_and_rejects_config_changes(monkeypatch):
    from realopen_lens.runtime import query_server
    from realopen_lens.encoders import onnx
    from realopen_lens.models import manager

    loads = []

    class Encoder:
        def __init__(self, graph, tokenizer, kind, threads):
            loads.append((kind, str(graph)))
            self.kind = kind

        def encode(self, texts):
            return np.ones((1, 768 if self.kind == "siglip2" else 384), dtype=np.float32)

    monkeypatch.setattr(query_server, "enforce", lambda: None)
    monkeypatch.setattr(onnx, "TextEncoder", Encoder)
    monkeypatch.setattr(manager, "artifact_path", lambda root, kind, key: Path(key))
    requests = [
        {"query": "hello", "kind": k, "models": "models", "threads": 2, "siglip_precision": "fp32"}
        for k in ("siglip2", "siglip2", "e5")
    ]
    requests.append(dict(requests[0], siglip_precision="int8"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(map(json.dumps, requests)) + "\n"))
    monkeypatch.setattr(sys, "stdout", output)
    assert query_server.serve() == 1
    assert loads == [("siglip2", "text_fp32"), ("e5", "text_int8")]
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == 4 and "configuration changed" in responses[-1]["error"]
    assert not any(t.name.startswith("lens-query-") for t in threading.enumerate())


def test_interactive_cli_continues_after_error_and_closes_session(monkeypatch):
    from typer.testing import CliRunner
    from realopen_lens.cli import app
    from realopen_lens import search

    calls = []

    class Session:
        def __init__(self, **options):
            calls.append("open")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("close")

        def search(self, query, **options):
            calls.append(query)
            if query == "bad":
                raise TimeoutError("deadline")
            return []

    monkeypatch.setattr(search, "SearchSession", Session)
    result = CliRunner().invoke(app, ["search", "bad", "--interactive"], input="next\n\n")
    assert result.exit_code == 1
    assert "deadline" in result.output and "[]" in result.output
    assert calls == ["open", "bad", "next", "close"]


def test_query_cohort_is_balanced_deterministic_and_keeps_translations():
    from benchmarks.experiments.resident_inference import select_queries

    bundle = {
        "queries": [
            {
                "id": f"{kind}-{family}-{language}",
                "kind": kind,
                "family": f"{kind}-{family}",
                "language": language,
            }
            for kind in ("photos", "documents", "diagrams", "screenshots")
            for family in range(4)
            for language in ("en", "fr", "ar")
        ]
    }
    chosen = select_queries(bundle, 2, 20261020)
    assert chosen == select_queries(bundle, 2, 20261020)
    assert len(chosen) == 24
    for kind in ("photos", "documents", "diagrams", "screenshots"):
        rows = [q for q in chosen if q["kind"] == kind]
        assert len(rows) == 6 and {q["language"] for q in rows} == {"en", "fr", "ar"}
    with pytest.raises(ValueError):
        select_queries(bundle, 5, 20261020)


def test_keyboard_interrupt_reaps_owned_worker(encoder, monkeypatch):
    client, root, _, _ = encoder
    encode(client, root)
    pid = client.pid

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(client._queue, "get", interrupted)
    with pytest.raises(KeyboardInterrupt):
        encode(client, root)
    assert client.pid is None and not psutil.pid_exists(pid)


def test_interrupted_benchmark_archives_partial_resident_sequence_and_rewarms(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from benchmarks.experiments import resident_inference as experiment

    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    query = {"id": "q", "family": "q", "text": "query", "kind": "photos"}
    monkeypatch.setattr(experiment, "load", lambda *a: ({}, {"queries": [query]}))
    monkeypatch.setattr(
        experiment,
        "snapshot",
        lambda index, *a, **k: {
            "pipeline_spec": {"siglip_precision": index.name},
            "generation": index.name,
        },
    )
    calls = []

    class Session:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(experiment, "SearchSession", Session)
    interrupt = True

    def measure(q, index, collection, models, threads, session=None):
        calls.append((index.name, session is not None))
        if interrupt and len(calls) == 3:
            raise KeyboardInterrupt
        return {
            "query_id": "q",
            "query": q,
            "status": "ok",
            "trace": {},
            "results": [],
            "seconds": 0.1,
            "peak_tree_rss_mib": 1,
            "profile": {},
            "metrics": {},
            "memory_access_errors": 0,
            "worker_pid": None,
            "resident_worker_rss_mib": None,
        }

    monkeypatch.setattr(experiment, "measure", measure)
    args = SimpleNamespace(
        corpus=reference,
        judgments=reference,
        int8_index=tmp_path / "int8",
        fp32_index=tmp_path / "fp32",
        collection="personal",
        models=tmp_path,
        threads=2,
        repeats=1,
        families_per_kind=1,
        seed=1,
        corpus_prefix=Path("."),
        allow_extra_images=False,
        output=tmp_path / "run",
    )
    with pytest.raises(KeyboardInterrupt):
        experiment.run(args)
    interrupt = False
    calls.clear()
    report = experiment.run(args)
    assert report["all_pairs_identical"] and report["errors"] == 0
    assert calls.count(("int8", False)) == 0  # Completed single-shot query resumed.
    assert calls.count(("int8", True)) == 2  # Startup and warm query rerun together.
    archived = json.loads((args.output / "int8-interrupted-001.json").read_text(encoding="utf-8"))
    assert archived["requests"][0]["phase"] == "resident_cold"
