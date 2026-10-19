"""Local OCR resource and networking contracts."""

import json
import subprocess
import sys


def test_loopback_exception_keeps_external_and_other_local_ports_blocked():
    code = """
import sys,json
from realopen_lens.offline import enforce
enforce(loopback_port=11434)
sys.audit("socket.getaddrinfo", "127.0.0.1", 11434, 0, 0, 0)
sys.audit("socket.connect", None, ("127.0.0.1", 11434))
blocked=0
for event,args in [("socket.getaddrinfo",("example.com",443,0,0,0)),
                   ("socket.connect",(None,("127.0.0.1",11435))),
                   ("socket.connect",(None,("192.0.2.1",11434))),
                   ("socket.sendto",(None,b"x",("127.0.0.1",11434)))]:
    try:sys.audit(event,*args)
    except PermissionError:blocked+=1
print(json.dumps(blocked))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == 4



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
