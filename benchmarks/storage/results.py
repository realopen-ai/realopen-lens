"""Durable benchmark stages, vector records, and attempt history."""

import json
import sqlite3
from pathlib import Path
import numpy as np


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS stages (
            image_id TEXT, stage TEXT, payload TEXT NOT NULL,
            PRIMARY KEY(image_id, stage));
        CREATE TABLE IF NOT EXISTS stage_history (
            attempt INTEGER PRIMARY KEY, image_id TEXT, stage TEXT, payload TEXT,
            archived_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS vectors (
            image_id TEXT, stage TEXT, shape TEXT, data BLOB,
            PRIMARY KEY(image_id, stage));
    """)
    return db


def save_stage(
    db: sqlite3.Connection,
    image_id: str,
    stage: str,
    payload: dict,
    vector: np.ndarray | None = None,
) -> None:
    with db:
        db.execute(
            "INSERT INTO stage_history(image_id,stage,payload) "
            "SELECT image_id,stage,payload FROM stages WHERE image_id=? AND stage=?",
            (image_id, stage),
        )
        db.execute("DELETE FROM vectors WHERE image_id=? AND stage=?", (image_id, stage))
        if payload.get("status") != "ok":
            vector = None
        if vector is not None:
            vector = np.asarray(vector, dtype=np.float32)
            db.execute(
                "INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                (image_id, stage, json.dumps(vector.shape), vector.tobytes()),
            )
        db.execute(
            "INSERT OR REPLACE INTO stages VALUES (?,?,?)",
            (image_id, stage, json.dumps(payload, ensure_ascii=False)),
        )


def reset_errors(db: sqlite3.Connection) -> None:
    with db:
        db.execute(
            "INSERT INTO stage_history(image_id,stage,payload) "
            "SELECT image_id,stage,payload FROM stages WHERE "
            "json_extract(payload,'$.status') IN ('error','timeout','blocked')"
        )
        db.execute(
            "DELETE FROM vectors WHERE EXISTS (SELECT 1 FROM stages "
            "WHERE stages.image_id=vectors.image_id AND stages.stage=vectors.stage "
            "AND json_extract(payload,'$.status') IN ('error','timeout','blocked'))"
        )
        db.execute(
            "DELETE FROM stages WHERE json_extract(payload,'$.status') "
            "IN ('error','timeout','blocked')"
        )


def load_vectors(db: sqlite3.Connection, stage: str) -> tuple[list[str], np.ndarray]:
    rows = db.execute(
        "SELECT image_id,shape,data FROM vectors WHERE stage=? ORDER BY image_id", (stage,)
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    ids = []
    arrays = []
    for image_id, shape, data in rows:
        array = np.frombuffer(data, dtype=np.float32).reshape(json.loads(shape))
        arrays.append(array)
        ids.extend([image_id] * len(array))
    return ids, np.concatenate(arrays)
