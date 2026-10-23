"""Frozen entry point also dispatches the engine's isolated Python workers."""

import runpy
import sys

WORKERS = {
    "realopen_lens.runtime.worker",
    "realopen_lens.runtime.query_server",
}


def main():
    if sys.platform == "win32":
        # Frozen worker pipes and CLI JSON use UTF-8 independently of the user's
        # Windows code page. Reconfigure each child too, before reading requests.
        for stream in (sys.stdin, sys.stdout, sys.stderr):
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["--check-gui"]:
        import json
        from realopen_lens.desktop.diagnostics import inspect_gui

        print(json.dumps(inspect_gui()))
        return
    if sys.argv[1:] == ["--check-bundle"]:
        import json
        from realopen_lens.desktop.diagnostics import inspect_bundle

        print(json.dumps(inspect_bundle()))
        return
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
