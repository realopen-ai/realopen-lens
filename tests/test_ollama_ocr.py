"""Local OCR resource and networking contracts."""


def test_external_server_memory_counts_each_process_once(monkeypatch):
    from types import SimpleNamespace
    from benchmarks.resources import MemoryMonitor

    class Process:
        def __init__(self, pid=None):
            self.pid = pid or 1

        def children(self, recursive=True):
            return [Process(2), Process(3)] if self.pid == 1 else [Process(3)]

        def memory_info(self):
            return SimpleNamespace(rss=self.pid * 100)

    monkeypatch.setattr("benchmarks.resources.psutil.Process", Process)
    monitor = MemoryMonitor(external_pids=(2,))
    monitor._sample()
    assert monitor.peaks["startup"] == 600
    assert monitor.samples == 1
