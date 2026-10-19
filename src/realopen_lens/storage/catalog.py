"""SQLite catalog and immutable analysis-cache records for local collections."""

from contextlib import contextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

STAGES = ("metadata", "ocr", "vision", "semantic")
REQUIRED_STAGES = ("metadata", "vision")


def now() -> str:
    return datetime.now(UTC).isoformat()


def default_directory() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "RealOpenLens"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/RealOpenLens"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "realopen-lens"


def identifier() -> str:
    return uuid4().hex


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Index artifact escapes the index directory")
    return path


class Catalog:
    def __init__(self, root: Path, *, readonly: bool = False):
        self.root = root.resolve()
        path = self.root / "catalog.sqlite3"
        if readonly and not path.is_file():
            raise ValueError("No collection catalog found; run 'lens index' first")
        if not readonly:
            self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"), uri=True, timeout=30
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2) or (readonly and version == 0):
            self.close()
            raise ValueError("Unsupported collection catalog version")
        if not readonly:
            if (
                version == 0
                and self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone()
            ):
                self.close()
                raise ValueError("Refusing to initialize an unrelated SQLite database")
            self.db.execute("PRAGMA journal_mode=WAL")
            if version == 1:
                with self.db:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.execute("ALTER TABLE analyses ADD COLUMN job_id TEXT")
                    self.db.execute("PRAGMA user_version=2")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, root TEXT NOT NULL,
                    active_generation TEXT, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id),
                    status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
                    config TEXT NOT NULL, phase TEXT, current_image TEXT, item_started REAL,
                    worker_pid INTEGER, worker_created REAL, error TEXT, result TEXT);
                CREATE TABLE IF NOT EXISTS files (
                    job_id TEXT NOT NULL REFERENCES jobs(id), path TEXT NOT NULL,
                    image_id TEXT, signature TEXT, status TEXT NOT NULL, error TEXT,
                    PRIMARY KEY(job_id,path));
                CREATE INDEX IF NOT EXISTS files_image ON files(job_id,image_id);
                CREATE TABLE IF NOT EXISTS analyses (
                    pipeline TEXT NOT NULL, image_id TEXT NOT NULL, stage TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL, artifact TEXT, checksum TEXT,
                    job_id TEXT,
                    PRIMARY KEY(pipeline,image_id,stage));
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id),
                    job_id TEXT NOT NULL REFERENCES jobs(id), manifest TEXT NOT NULL,
                    checksum TEXT NOT NULL, created_at TEXT NOT NULL);
                PRAGMA user_version=2;
            """)

    def close(self):
        self.db.close()

    def collection(self, name: str) -> dict:
        row = self.db.execute("SELECT * FROM collections WHERE name=?", (name,)).fetchone()
        if row is None:
            raise ValueError(f"Collection {name!r} not found; run 'lens index' first")
        return dict(row)

    def ensure_collection(self, name: str, root: Path) -> dict:
        if not name.strip() or len(name) > 128:
            raise ValueError("Collection name must contain 1–128 characters")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO collections VALUES (?,?,?,?,?)",
                (identifier(), name, str(root), None, now()),
            )
        collection = self.collection(name)
        if collection["root"] != str(root):
            raise ValueError("Collection already belongs to a different root; choose another name")
        return collection

    def job(self, job_id: str) -> dict:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Index job not found")
        return dict(row)

    def create_job(self, collection_id: str, config: dict) -> str:
        job_id = identifier()
        with self.db:
            self.db.execute(
                "INSERT INTO jobs(id,collection_id,status,started_at,config) " "VALUES (?,?,?,?,?)",
                (job_id, collection_id, "running", now(), json.dumps(config, sort_keys=True)),
            )
        return job_id

    def files(self, job_id: str) -> list[dict]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM files WHERE job_id=? ORDER BY path", (job_id,)
            )
        ]

    def add_file(self, job_id, path, image_id, signature, error=None):
        with self.db:
            self.db.execute(
                "INSERT INTO files VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    path,
                    image_id,
                    json.dumps(signature),
                    "error" if error else "pending",
                    error,
                ),
            )

    def analysis(self, pipeline: str, image_id: str, stage: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM analyses WHERE pipeline=? AND image_id=? AND stage=?",
            (pipeline, image_id, stage),
        ).fetchone()
        return dict(row) if row else None

    def save_analysis(
        self, pipeline, image_id, stage, payload, artifact=None, checksum=None, *, job_id=None
    ):
        with self.db:
            if stage == "ocr":
                # A retried OCR result can change its text; old E5 rows must not survive it.
                self.db.execute(
                    "DELETE FROM analyses WHERE pipeline=? AND image_id=? AND stage='semantic'",
                    (pipeline, image_id),
                )
            self.db.execute(
                "INSERT OR REPLACE INTO analyses "
                "(pipeline,image_id,stage,status,payload,artifact,checksum,job_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    pipeline,
                    image_id,
                    stage,
                    payload["status"],
                    json.dumps(payload),
                    artifact,
                    checksum,
                    job_id,
                ),
            )

    def fail_image(self, job_id: str, image_id: str, message: str):
        with self.db:
            self.db.execute(
                "UPDATE files SET status='error',error=? WHERE job_id=? AND image_id=?",
                (message[:1500], job_id, image_id),
            )

    def fail_stage(self, job_id: str, image_id: str, stage: str, message: str):
        if stage in REQUIRED_STAGES:
            self.fail_image(job_id, image_id, f"{stage}: {message}")
        else:
            with self.db:
                self.db.execute(
                    "UPDATE files SET status='degraded',error=? "
                    "WHERE job_id=? AND image_id=? AND status!='error'",
                    (f"{stage}: {message}"[:1500], job_id, image_id),
                )

    def heartbeat(self, job_id: str, stage: str, image_id: str | None, started: float | None):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET phase=?,current_image=?,item_started=? WHERE id=?",
                (stage, image_id, started, job_id),
            )

    def worker(self, job_id: str, pid: int | None, created: float | None):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET worker_pid=?,worker_created=? WHERE id=?", (pid, created, job_id)
            )

    def finish(self, job_id: str, status: str, *, result=None, error=None):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status=?,finished_at=?,error=?,result=?,"
                "current_image=NULL,item_started=NULL WHERE id=?",
                (status, now(), error, json.dumps(result) if result else None, job_id),
            )

    def snapshot(self, name: str) -> tuple[dict, dict]:
        with self.transaction():
            collection = self.collection(name)
            row = self.db.execute(
                "SELECT * FROM generations WHERE id=?", (collection["active_generation"],)
            ).fetchone()
            if row is None:
                raise ValueError("Collection has no published index yet")
            return collection, dict(row)

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def publish(self, collection_id, job_id, generation_id, manifest, checksum):
        with self.db:
            self.db.execute(
                "INSERT INTO generations VALUES (?,?,?,?,?,?)",
                (generation_id, collection_id, job_id, manifest, checksum, now()),
            )
            self.db.execute(
                "UPDATE collections SET active_generation=? WHERE id=?",
                (generation_id, collection_id),
            )
