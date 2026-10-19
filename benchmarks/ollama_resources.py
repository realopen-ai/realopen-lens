"""Identify the already-running local Ollama server for external memory sampling."""

import psutil


def server_pids(expected_pid: int | None = None) -> tuple[int, ...]:
    matches = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        name = (process.info.get("name") or "").lower()
        arguments = process.info.get("cmdline") or []
        if (
            "ollama" in name
            and "serve" in arguments
            and (expected_pid is None or process.pid == expected_pid)
        ):
            matches.append(process.pid)
    if len(matches) != 1:
        raise ValueError("Expected one local Ollama server; cannot attribute its memory reliably")
    return tuple(matches)
