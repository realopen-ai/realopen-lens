"""Opt-in, low-priority folder monitoring using the existing observable index jobs."""

from copy import deepcopy
import json
import logging
from pathlib import Path
import threading
from time import monotonic, time
from typing import TYPE_CHECKING

from realopen_lens.api.collections import collections
from realopen_lens.indexing.changes import indexed, observed, source_identity
from realopen_lens.runtime.jobs import writer_lock
from realopen_lens.storage.artifacts import atomic_bytes
from realopen_lens.storage.catalog import identifier

if TYPE_CHECKING:
    from realopen_lens.api.tasks import TaskManager

logger = logging.getLogger(__name__)
SETTLE_SECONDS = 10


class FolderRefresh:
    def __init__(self, manager: "TaskManager"):
        self.manager = manager
        self.path = manager.root / "folder-refresh.json"
        self.settings: dict[str, dict] = {}
        self.states: dict[str, dict] = {}
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.load_error: str | None = None
        if self.path.is_file():
            try:
                self._load()
            except (OSError, ValueError, TypeError):
                self.settings.clear()
                self.load_error = (
                    "Automatic refresh settings could not be read. Re-enable to save new settings; "
                    "the previous file will be backed up."
                )
                logger.warning("Automatic refresh disabled: cannot read %s", self.path)

    def _bytes(self) -> bytes:
        with self.path.open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError(
                "Automatic refresh settings exceed size limit; "
                "move the settings file aside before enabling"
            )
        return data

    def _load(self):
        data = json.loads(self._bytes())
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or not isinstance(data.get("folders"), dict)
        ):
            raise ValueError("Invalid automatic refresh settings")
        for key, value in data["folders"].items():
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("enabled"), bool)
                or not isinstance(value.get("interval_seconds"), int)
                or not 30 <= value["interval_seconds"] <= 3600
                or not isinstance(value.get("identity"), list)
                or len(value["identity"]) != 2
                or not all(type(item) is int for item in value["identity"])
                or not isinstance(value.get("pause_reason", ""), str)
            ):
                raise ValueError("Invalid automatic refresh settings")
            self.settings[key] = value

    def _save(self):
        if self.load_error and self.path.is_file():
            atomic_bytes(
                self.path.with_name(f"folder-refresh.invalid-{identifier()}.json"), self._bytes()
            )
        atomic_bytes(
            self.path, json.dumps({"version": 1, "folders": self.settings}).encode("utf-8")
        )
        self.load_error = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="lens-folder-refresh", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def describe(self, collection_id: str) -> dict:
        with self.manager.lock:
            config = self.settings.get(collection_id, {})
            state = self.states.get(collection_id, {})
            return {
                "enabled": config.get("enabled", False),
                "interval_seconds": config.get("interval_seconds", 60),
                "state": (
                    self.manager.tasks[state["task_id"]]["status"]
                    if state.get("task_id") in self.manager.tasks and config.get("enabled")
                    else state.get("state", "waiting" if config.get("enabled") else "off")
                ),
                "error": self.load_error or state.get("error") or config.get("pause_reason"),
                "last_checked_at": state.get("last_checked_at"),
                "task_id": state.get("task_id"),
            }

    def configure(
        self, name: str, collection_id: str, enabled: bool, interval_seconds: int
    ) -> dict:
        from realopen_lens.organization.folders import checked
        from realopen_lens.storage.catalog import Catalog

        if not 30 <= interval_seconds <= 3600:
            raise ValueError("Automatic refresh interval must be 30–3600 seconds")
        with self.manager.lock:
            catalog = Catalog(self.manager.root, readonly=True)
            try:
                folder = checked(catalog, name, collection_id)
            finally:
                catalog.close()
            previous = self.settings.get(collection_id)
            config = {
                "enabled": enabled,
                "interval_seconds": interval_seconds,
                "identity": (
                    (
                        previous["identity"]
                        if previous and previous["enabled"]
                        else source_identity(Path(folder["root"]))
                    )
                    if enabled
                    else self.settings.get(collection_id, {}).get("identity", [0, 0])
                ),
            }
            previous_state = self.states.get(collection_id)
            self.settings[collection_id] = config
            self.states[collection_id] = {
                "state": "waiting" if enabled else "off",
                "next_check": monotonic(),
                "revision": object(),
            }
            try:
                self._save()
            except Exception:
                if previous is None:
                    self.settings.pop(collection_id, None)
                else:
                    self.settings[collection_id] = previous
                if previous_state is None:
                    self.states.pop(collection_id, None)
                else:
                    self.states[collection_id] = previous_state
                raise
            return self.describe(collection_id)

    def forget(self, collection_id: str):
        if collection_id in self.settings:
            self.settings.pop(collection_id)
            self.states.pop(collection_id, None)
            self._save()

    def cancelled(self, state: dict):
        if state.get("origin") == "automatic" and state.get("collection_id") in self.settings:
            key = state["collection_id"]
            self.settings[key].update(
                enabled=False,
                pause_reason="Paused after cancellation; enable automatic refresh to resume",
            )
            self.states.setdefault(key, {}).update(state="off")
            self._save()

    def finished(self, state: dict):
        if state.get("origin") != "automatic":
            return
        key = state["collection_id"]
        if key not in self.settings:
            return
        config = self.settings[key]
        current = self.states.setdefault(key, {})
        current.update(task_id=None, next_check=monotonic() + config["interval_seconds"])
        if state["status"] == "failed" and not self.manager.closed:
            config.update(
                enabled=False,
                pause_reason="Automatic refresh failed: " + (state.get("error") or "Unknown error"),
            )
            self._save()
        current["state"] = "waiting" if config["enabled"] else "off"

    def _loop(self):
        while not self.stop.wait(2):
            try:
                self.check_once()
            except Exception:
                logger.exception("Automatic folder refresh check failed")

    def check_once(self, *, now: float | None = None):
        """One bounded pass; public for deterministic scheduling/integration tests."""
        from realopen_lens.api.tasks import TERMINAL

        instant = monotonic() if now is None else now
        with self.manager.lock:
            if (
                self.manager.closed
                or self.stop.is_set()
                or not any(c["enabled"] for c in self.settings.values())
            ):
                return
            # Foreground work always gets the single inference worker first.
            if any(t["status"] not in TERMINAL for t in self.manager.tasks.values()):
                return
        for folder in collections(self.manager.root):
            instant = monotonic() if now is None else now
            key = folder["id"]
            with self.manager.lock:
                config = deepcopy(self.settings.get(key, {}))
                state = self.states.setdefault(key, {"revision": object()})
                if not config.get("enabled") or state.get("next_check", 0) > instant:
                    continue
                revision = state.get("revision")
                state.update(state="checking", last_checked_at=time())
            try:
                actual = observed(
                    Path(folder["root"]), self.manager.root, config["identity"], self.stop.is_set
                )
                baseline = indexed(self.manager.root, key)
                error = None
            except (OSError, ValueError) as exc:
                actual, baseline, error = None, None, str(exc)
            with self.manager.lock:
                if (
                    self.manager.closed
                    or self.stop.is_set()
                    or self.states.get(key) is not state
                    or state.get("revision") is not revision
                    or not self.settings.get(key, {}).get("enabled")
                ):
                    continue
                instant = monotonic() if now is None else now
                state.update(error=error, next_check=instant + config["interval_seconds"])
                if error:
                    state.update(state="unavailable", candidate=None)
                elif actual == baseline:
                    state.update(state="waiting", candidate=None)
                elif state.get("candidate") != actual:
                    state.update(
                        state="settling",
                        candidate=actual,
                        stable_since=instant,
                        next_check=instant + SETTLE_SECONDS,
                    )
                elif instant < state["stable_since"] + SETTLE_SECONDS:
                    state["state"] = "settling"
                elif any(t["status"] not in TERMINAL for t in self.manager.tasks.values()):
                    state["state"] = "waiting"
                else:
                    try:
                        # A CLI indexer may own the catalog without an API task.
                        with writer_lock(self.manager.root):
                            pass
                    except ValueError:
                        state.update(state="waiting", error="Another indexing job is running")
                        return
                    try:
                        task = self.manager.refresh_folder(
                            folder["name"], key, automatic=True, source_identity=config["identity"]
                        )
                    except ValueError as exc:
                        state.update(state="unavailable", error=str(exc))
                    else:
                        state.update(state="queued", task_id=task["id"], candidate=None)
                        logger.info(
                            "Automatic refresh queued for %r: %s", folder["name"], task["id"]
                        )
                    return  # At most one automatic index task per pass.
