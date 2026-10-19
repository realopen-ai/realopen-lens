"""Exercise native-worker timeout isolation using small real child processes."""

import numpy as np


from benchmarks.suite import connect, load_vectors, reset_errors, save_stage


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
