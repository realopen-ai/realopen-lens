"""Defense in depth for Python networking, not an OS network sandbox."""

import os
import sys


def enforce() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def audit(event: str, args: tuple) -> None:
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
            raise PermissionError("Network access is disabled during the benchmark")

    sys.addaudithook(audit)
