"""Separate, versioned SQLite store for opt-in searches and explicit judgments."""

from pathlib import Path
import json
import sqlite3

from realopen_lens.storage.catalog import identifier, now


class FeedbackStore:
    def __init__(self, root: Path, *, readonly: bool = False):
        self.path = root.resolve() / "feedback.sqlite3"
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"), uri=True, timeout=30
        )
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA foreign_keys=ON")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1) or (readonly and version == 0):
                raise ValueError("Unsupported feedback database version")
            if not readonly:
                if (
                    version == 0
                    and self.db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    ).fetchone()
                ):
                    raise ValueError("Refusing to initialize an unrelated feedback database")
                self.db.execute("PRAGMA journal_mode=WAL")
                self.db.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS searches (
                        id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                        collection_id TEXT NOT NULL, payload TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS judgments (
                        run_id TEXT NOT NULL REFERENCES searches(id), image_id TEXT NOT NULL,
                        label TEXT NOT NULL CHECK(label IN ('relevant','irrelevant')),
                        revision INTEGER NOT NULL, updated_at TEXT NOT NULL,
                        PRIMARY KEY(run_id,image_id));
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL REFERENCES searches(id), image_id TEXT NOT NULL,
                        label TEXT CHECK(label IN ('relevant','irrelevant')),
                        revision INTEGER NOT NULL, created_at TEXT NOT NULL);
                    CREATE INDEX IF NOT EXISTS event_pair ON events(run_id,image_id,id);
                    PRAGMA user_version=1;
                    COMMIT;
                """)
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def add_run(self, payload: dict) -> str:
        run_id = identifier()
        with self.db:
            self.db.execute(
                "INSERT INTO searches VALUES (?,?,?,?)",
                (
                    run_id,
                    now(),
                    payload["collection_id"],
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False),
                ),
            )
        return run_id

    def run(self, run_id: str) -> dict:
        row = self.db.execute("SELECT * FROM searches WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Saved search run not found")
        return {"run_id": row["id"], "created_at": row["created_at"], **json.loads(row["payload"])}

    def runs(self, limit: int) -> list[dict]:
        if not 1 <= limit <= 100:
            raise ValueError("Use 1–100 saved runs")
        return [
            {
                "run_id": row["id"],
                "created_at": row["created_at"],
                **{k: json.loads(row["payload"])[k] for k in ("collection", "query", "generation")},
            }
            for row in self.db.execute(
                "SELECT * FROM searches ORDER BY rowid DESC LIMIT ?", (limit,)
            )
        ]

    def judgments(self, run_id: str) -> list[dict]:
        self.run(run_id)
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM judgments WHERE run_id=? ORDER BY image_id", (run_id,)
            )
        ]

    def history(self, run_id: str) -> list[dict]:
        self.run(run_id)
        return [
            dict(row)
            for row in self.db.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))
        ]

    def judge(self, run_id: str, image_id: str, label: str | None) -> dict | None:
        if label is not None and label not in {"relevant", "irrelevant"}:
            raise ValueError("Label must be relevant or irrelevant")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            run = self.run(run_id)
            if image_id not in {r["image_id"] for r in run["results"]}:
                raise ValueError("Image was not returned by this saved search")
            current = self.db.execute(
                "SELECT * FROM judgments WHERE run_id=? AND image_id=?", (run_id, image_id)
            ).fetchone()
            if (current is None and label is None) or (current and current["label"] == label):
                return dict(current) if current else None
            previous = self.db.execute(
                "SELECT MAX(revision) FROM events WHERE run_id=? AND image_id=?", (run_id, image_id)
            ).fetchone()[0]
            revision, timestamp = (previous or 0) + 1, now()
            self.db.execute(
                "INSERT INTO events(run_id,image_id,label,revision,created_at) "
                "VALUES (?,?,?,?,?)",
                (run_id, image_id, label, revision, timestamp),
            )
            self.db.execute(
                "DELETE FROM judgments WHERE run_id=? AND image_id=?", (run_id, image_id)
            )
            if label is None:
                return None
            self.db.execute(
                "INSERT INTO judgments VALUES (?,?,?,?,?)",
                (run_id, image_id, label, revision, timestamp),
            )
            return {
                "run_id": run_id,
                "image_id": image_id,
                "label": label,
                "revision": revision,
                "updated_at": timestamp,
            }
