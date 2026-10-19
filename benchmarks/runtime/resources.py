"""Sample process-tree RSS, including OCR subprocesses, throughout a benchmark."""

import threading
from collections import defaultdict
from typing import Self
import psutil
from pathlib import Path
from typing import Any
import json


class MemoryMonitor:
    def __init__(self, external_pids: tuple[int, ...] = ()) -> None:
        self.external_pids = external_pids
        self.phase = "startup"
        self.peaks: dict[str, int] = defaultdict(int)
        self.samples = 0
        self.access_errors = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        parent = psutil.Process()
        total = 0
        try:
            processes = [parent, *parent.children(recursive=True)]
        except (psutil.AccessDenied, OSError):
            self.access_errors += 1
            processes = [parent]
        for pid in self.external_pids:
            try:
                external = psutil.Process(pid)
                processes.extend([external, *external.children(recursive=True)])
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, OSError):
                self.access_errors += 1
        for process in {p.pid: p for p in processes}.values():
            try:
                total += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, OSError):
                self.access_errors += 1
        self.peaks[self.phase] = max(self.peaks[self.phase], total)
        self.samples += 1

    def _run(self) -> None:
        while not self.stop.is_set():
            self._sample()
            self.stop.wait(0.02)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        self.thread.join()
        self._sample()


def write_resources(output: Path, memory: MemoryMonitor) -> None:
    current: dict[str, Any] = {
        "peak_tree_rss_mib": max(memory.peaks.values(), default=0) / 2**20,
        "phase_peak_mib": {key: value / 2**20 for key, value in memory.peaks.items()},
        "samples": memory.samples,
        "access_errors": memory.access_errors,
        "sample_interval_ms": 20,
        "scope": "Coordinator and children, including model loading and evaluation",
        "limitations": "Shared pages may count twice; spikes between samples may be missed",
    }
    number = len(list(output.glob("resources-attempt-*.json"))) + 1
    attempt = output / f"resources-attempt-{number:03d}.json"
    attempt.write_text(json.dumps(current, indent=2) + "\n")
    destination = output / "resources.json"
    previous = json.loads(destination.read_text()) if destination.exists() else {}
    peaks = previous.get("phase_peak_mib", {})
    for key, value in current["phase_peak_mib"].items():
        peaks[key] = max(peaks.get(key, 0), value)
    merged = {
        **current,
        "phase_peak_mib": peaks,
        "peak_tree_rss_mib": max(
            current["peak_tree_rss_mib"], previous.get("peak_tree_rss_mib", 0)
        ),
        "access_errors": current["access_errors"] + previous.get("access_errors", 0),
        "samples": current["samples"] + previous.get("samples", 0),
        "invocations": previous.get("invocations", []) + [attempt.name],
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=2) + "\n")
    temporary.replace(destination)
