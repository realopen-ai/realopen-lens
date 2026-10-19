"""Defense in depth for Python networking, not an OS network sandbox."""

import os
import sys


def enforce(loopback_port: int | None = None) -> None:
    """Deny networking, optionally allowing TCP to one numeric loopback endpoint."""
    if loopback_port is not None and not 1 <= loopback_port <= 65535:
        raise ValueError("Invalid loopback port")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def audit(event: str, args: tuple) -> None:
        if loopback_port is not None:
            address = (
                args[:2]
                if event == "socket.getaddrinfo"
                else (args[1] if event in {"socket.connect", "socket.connect_ex"} else ())
            )
            if isinstance(address, tuple) and address[:2] == ("127.0.0.1", loopback_port):
                return
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
            raise PermissionError("Network access is disabled during local inference")

    sys.addaudithook(audit)
