"""Exercise native-worker timeout isolation using small real child processes."""

import json


import os


import sys


import numpy as np


import pytest


from benchmarks.execution import run_stage


from benchmarks.suite import connect, load_vectors, reset_errors, save_stage


def test_hung_item_is_recorded_and_later_items_still_run(tmp_path):
    path = tmp_path / "results.sqlite"
    db = connect(path)
    code = f"""
import sqlite3,json,time
connection=sqlite3.connect({str(path)!r})
if not connection.execute("SELECT 1 FROM stages WHERE image_id='bad'").fetchone():
    payload=json.dumps({{'id':'bad','started_at':time.monotonic()}})
    connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",('ocr_current',payload))
    connection.commit()
    time.sleep(10)
payload=json.dumps({{'status':'ok'}})
connection.execute("INSERT OR REPLACE INTO stages VALUES (?,?,?)",('good','ocr',payload))
connection.commit()
"""
    events = run_stage(
        [sys.executable, "-c", code],
        db,
        "ocr",
        dict(os.environ),
        tmp_path / "worker.log",
        lambda key, payload: save_stage(db, key, "ocr", payload),
        image_timeout=0.15,
        startup_timeout=3,
        stage_timeout=5,
        poll_interval=0.02,
    )
    records = {
        key: json.loads(value) for key, value in db.execute("SELECT image_id,payload FROM stages")
    }
    assert records["bad"]["status"] == "timeout"
    assert records["good"]["status"] == "ok"
    assert len(events) == 1
    db.close()


def test_worker_startup_failure_does_not_fabricate_image_results(tmp_path):
    db = connect(tmp_path / "results.sqlite")
    with pytest.raises(RuntimeError, match="before an item"):
        run_stage(
            [sys.executable, "-c", "raise SystemExit(7)"],
            db,
            "vision",
            dict(os.environ),
            tmp_path / "worker.log",
            lambda key, payload: save_stage(db, key, "vision", payload),
            startup_timeout=2,
            stage_timeout=3,
            poll_interval=0.01,
        )
    assert db.execute("SELECT COUNT(*) FROM stages").fetchone() == (0,)
    db.close()


def test_retry_invalidates_blocked_dependents_and_failed_vectors(tmp_path):
    db = connect(tmp_path / "results.sqlite")
    save_stage(db, "a", "vision", {"status": "ok"}, np.array([[1.0, 2.0]]))
    save_stage(db, "a", "ocr", {"status": "timeout"})
    save_stage(db, "a", "semantic", {"status": "blocked"})
    save_stage(db, "bad", "vision", {"status": "error"}, np.array([[1.0, 2.0]]))
    assert load_vectors(db, "vision")[0] == ["a"]
    reset_errors(db)
    assert db.execute("SELECT image_id,stage FROM stages").fetchall() == [("a", "vision")]
    assert db.execute("SELECT COUNT(*) FROM stage_history").fetchone() == (3,)
    assert load_vectors(db, "vision")[0] == ["a"]
    db.close()
