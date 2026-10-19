"""Sample process-tree RSS, including OCR subprocesses, throughout a benchmark."""

import threading
from collections import defaultdict
from typing import Self

import psutil


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
