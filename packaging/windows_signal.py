"""Send Ctrl+Break even when the CI runner has no console attached."""

import ctypes
import sys


def send(pid: int):
    if sys.platform != "win32":
        raise ValueError("Windows console signals require Windows")
    # Run in a short-lived helper: detaching must not affect the smoke runner's
    # own terminal. The target is a process group created by that runner.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.FreeConsole()
    if not kernel.AttachConsole(pid):
        raise OSError(ctypes.get_last_error(), "Cannot attach to the owned server console")
    try:
        if not kernel.GenerateConsoleCtrlEvent(1, pid):  # CTRL_BREAK_EVENT
            raise OSError(ctypes.get_last_error(), "Cannot signal the owned server group")
    finally:
        kernel.FreeConsole()


if __name__ == "__main__":
    send(int(sys.argv[1]))
