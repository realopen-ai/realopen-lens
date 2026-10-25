"""Separate, versioned SQLite store for opt-in searches and explicit judgments."""

from pathlib import Path
import json
import hashlib
import sqlite3
import unicodedata

from realopen_lens.storage.catalog import identifier, now

MEMORY_POLICY = "exact-query-v1"


def normalize_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def memory_scope(scope: str, interpretation: dict | None) -> str:
    rules = [
        rule
        for rule in (interpretation or {}).get("constraints", [])
        if not (rule["kind"] == "screenshot" and rule.get("policy") == "markers_and_visual")
    ]
    if not rules:
        return scope
    values = []
    for rule in rules:
        value = {key: item for key, item in rule.items() if key != "label"}
        if value["kind"] in {"camera", "folder"}:
            value["value"] = normalize_query(value["value"])
        values.append(value)
    payload = json.dumps(
        sorted(values, key=lambda item: json.dumps(item, sort_keys=True)), sort_keys=True
    )
    return scope + ":metadata-v1:" + hashlib.sha256(payload.encode()).hexdigest()


def memory_key(payload: dict) -> tuple[str, str, str] | None:
    query = payload.get("query", {})
    if query.get("kind") != "text" or not isinstance(query.get("text"), str):
        return None
    return (
        payload["collection_id"],
        normalize_query(query["text"]),
        memory_scope(
            (payload.get("scope") or {}).get("id", ""),
            payload.get("ranking", {}).get("query_interpretation"),
        ),
    )


class FeedbackConflict(ValueError):
    pass


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
            if not readonly:
                self.db.execute("BEGIN IMMEDIATE")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2) or (readonly and version == 0):
                raise ValueError("Unsupported feedback database version")
            self.version = version
            if not readonly:
                if version == 0:
                    if self.db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE "
                        "'sqlite_%'"
                    ).fetchone():
                        raise ValueError("Refusing to initialize an unrelated feedback database")
                    for statement in (
                        "CREATE TABLE searches (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                        "collection_id TEXT NOT NULL, payload TEXT NOT NULL)",
                        "CREATE TABLE judgments (run_id TEXT NOT NULL REFERENCES searches(id), "
                        "image_id TEXT NOT NULL, label TEXT NOT NULL CHECK(label IN "
                        "('relevant','irrelevant')), "
                        "revision INTEGER NOT NULL, updated_at TEXT NOT NULL, PRIMARY "
                        "KEY(run_id,image_id))",
                        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "run_id TEXT NOT NULL REFERENCES searches(id), image_id TEXT NOT NULL, "
                        "label TEXT CHECK(label IN ('relevant','irrelevant')), revision "
                        "INTEGER NOT NULL, "
                        "created_at TEXT NOT NULL)",
                        "CREATE INDEX event_pair ON events(run_id,image_id,id)",
                    ):
                        self.db.execute(statement)
                if version < 2:
                    self.db.execute(
                        "CREATE TABLE memory_runs (run_id TEXT PRIMARY KEY REFERENCES "
                        "searches(id), "
                        "collection_id TEXT NOT NULL, query_key TEXT NOT NULL, scope_key TEXT "
                        "NOT NULL)"
                    )
                    self.db.execute(
                        "CREATE TABLE query_memory (collection_id TEXT NOT NULL, query_key "
                        "TEXT NOT NULL, "
                        "scope_key TEXT NOT NULL, image_id TEXT NOT NULL, label TEXT, "
                        "event_id INTEGER NOT NULL REFERENCES events(id), "
                        "run_id TEXT NOT NULL REFERENCES searches(id), "
                        "PRIMARY KEY(collection_id,query_key,scope_key,image_id))"
                    )
                    for row in self.db.execute("SELECT id,payload FROM searches").fetchall():
                        self._index_run(row["id"], json.loads(row["payload"]))
                    self.db.execute(
                        "INSERT OR REPLACE INTO query_memory "
                        "SELECT "
                        "m.collection_id,m.query_key,m.scope_key,e.image_id,e.label,e.id,e.run_id "
                        "FROM events e JOIN memory_runs m ON m.run_id=e.run_id ORDER BY e.id"
                    )
                    self.db.execute("PRAGMA user_version=2")
                self.db.commit()
                self.version = 2
                try:
                    self.db.execute("PRAGMA journal_mode=WAL")
                except sqlite3.OperationalError as error:
                    if error.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                        raise
        except BaseException:
            self.db.rollback()
            self.db.close()
            raise

    def _index_run(self, run_id: str, payload: dict):
        key = memory_key(payload)
        if key is not None:
            self.db.execute("INSERT INTO memory_runs VALUES (?,?,?,?)", (run_id, *key))

    def memory(self, collection_id: str, query: str, scope_key: str = "") -> list[dict]:
        """Read v2's indexed projection or v1 events without migrating on a read."""
        key = (collection_id, normalize_query(query), scope_key)
        if self.version == 2:
            return [
                dict(row)
                for row in self.db.execute(
                    "SELECT image_id,label,event_id,run_id FROM query_memory "
                    "WHERE collection_id=? AND query_key=? AND scope_key=? ORDER BY event_id",
                    key,
                )
            ]
        run_ids = []
        for row in self.db.execute(
            "SELECT id,payload FROM searches WHERE collection_id=?", (collection_id,)
        ):
            if memory_key(json.loads(row["payload"])) == key:
                run_ids.append(row["id"])
        latest: dict[str, dict] = {}
        for offset in range(0, len(run_ids), 400):
            batch = run_ids[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            for row in self.db.execute(
                "SELECT image_id,label,id AS event_id,run_id FROM events WHERE run_id IN ("
                + placeholders
                + ") ORDER BY id",
                batch,
            ):
                value = dict(row)
                if value["event_id"] > latest.get(value["image_id"], {}).get("event_id", 0):
                    latest[value["image_id"]] = value
        return sorted(latest.values(), key=lambda row: row["event_id"])

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
            self._index_run(run_id, payload)
        return run_id

    def run(self, run_id: str) -> dict:
        row = self.db.execute("SELECT * FROM searches WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Saved search run not found")
        return {**json.loads(row["payload"]), "run_id": row["id"], "created_at": row["created_at"]}

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
        return self.change(run_id, image_id, label)[0]

    def change(
        self,
        run_id: str,
        image_id: str,
        label: str | None,
        *,
        aliases: tuple[str, ...] = (),
        expected_event_id: int | None = None,
    ) -> tuple[dict | None, dict | None]:
        if label is not None and label not in {"relevant", "irrelevant"}:
            raise ValueError("Label must be relevant or irrelevant")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            run = self.run(run_id)
            if image_id not in {r["image_id"] for r in run["results"]}:
                raise ValueError("Image was not returned by this saved search")
            key = memory_key(run)
            candidates = self.memory(key[0], run["query"]["text"], key[2]) if key else []
            matching = [row for row in candidates if row["image_id"] in {*aliases, image_id}]
            effective = max(matching, key=lambda row: row["event_id"], default=None)
            if (
                expected_event_id is not None
                and (effective or {}).get("event_id", 0) != expected_event_id
            ):
                raise FeedbackConflict("Feedback changed; refresh before applying this action")
            current = self.db.execute(
                "SELECT * FROM judgments WHERE run_id=? AND image_id=?", (run_id, image_id)
            ).fetchone()
            same_label = (current is None and label is None) or (
                current and current["label"] == label
            )
            if same_label and (key is None or (effective or {}).get("label") == label):
                return (dict(current) if current else None), effective
            previous = self.db.execute(
                "SELECT MAX(revision) FROM events WHERE run_id=? AND image_id=?", (run_id, image_id)
            ).fetchone()[0]
            revision, timestamp = (previous or 0) + 1, now()
            event = self.db.execute(
                "INSERT INTO events(run_id,image_id,label,revision,created_at) VALUES (?,?,?,?,?)",
                (run_id, image_id, label, revision, timestamp),
            )
            if key is not None:
                self.db.execute(
                    "INSERT OR REPLACE INTO query_memory VALUES (?,?,?,?,?,?,?)",
                    (*key, image_id, label, event.lastrowid, run_id),
                )
                effective = {
                    "image_id": image_id,
                    "label": label,
                    "event_id": event.lastrowid,
                    "run_id": run_id,
                }
            self.db.execute(
                "DELETE FROM judgments WHERE run_id=? AND image_id=?", (run_id, image_id)
            )
            if label is None:
                return None, effective
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
            }, effective
