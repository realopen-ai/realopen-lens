"""Validate Uvicorn cleanup separately from the platform's signal exit status."""

import re
import signal


def verify_shutdown(returncode: int, log: str, *, windows: bool):
    # Uvicorn restores and re-raises the captured signal after cleanup. Windows
    # CRT SIG_DFL exits with 3; a console control handler can instead return
    # STATUS_CONTROL_C_EXIT. Neither code alone proves successful shutdown.
    expected = (0, 3, 3221225786, -1073741510) if windows else (0, -signal.SIGTERM)
    if returncode not in expected:
        raise ValueError(f"Frozen gallery exited unexpectedly: {returncode}")
    if not all(
        marker in log
        for marker in (
            "Application shutdown complete.",
            "Finished server process [",
        )
    ):
        raise ValueError("Frozen gallery shutdown did not complete")
    if "Traceback (most recent call last)" in log or re.search(r"(?m)^(ERROR|CRITICAL):", log):
        raise ValueError("Frozen gallery logged an error during acceptance")
