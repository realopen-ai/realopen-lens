"""Frozen entry point also dispatches the engine's isolated Python workers."""

import runpy
import sys

WORKERS = {
    "realopen_lens.runtime.worker",
    "realopen_lens.runtime.query_server",
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from realopen_lens.cli import app

        sys.argv.pop(1)
        app()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "-m":
        if len(sys.argv) < 3 or sys.argv[2] not in WORKERS:
            raise SystemExit("Unsupported Lens worker")
        module = sys.argv[2]
        sys.argv = sys.argv[2:]
        runpy.run_module(module, run_name="__main__")
        return
    from realopen_lens.desktop.launcher import main as launch

    launch()


if __name__ == "__main__":
    main()
