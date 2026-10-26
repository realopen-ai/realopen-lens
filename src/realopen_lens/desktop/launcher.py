"""Desktop lifecycle controller with no shell commands or implicit downloads."""

import json
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import re
import socket
import sys
import threading
import webbrowser
from typing import Any

from realopen_lens.runtime.precision import memory_bytes, recommended_precision

from realopen_lens.storage.catalog import default_directory
from realopen_lens.workspace.identity import metadata, models_directory


def settings(root: Path) -> dict:
    path = root / "desktop.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or (
                "models" in value
                and (
                    not isinstance(value["models"], str) or not Path(value["models"]).is_absolute()
                )
            )
            or (
                "ocr_languages" in value
                and (
                    not isinstance(value["ocr_languages"], str)
                    or not re.fullmatch(
                        r"(?:[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*)?", value["ocr_languages"]
                    )
                )
            )
            or (
                "siglip_precision" in value
                and (
                    not isinstance(value["siglip_precision"], str)
                    or value["siglip_precision"] not in {"int8", "fp32"}
                )
            )
        ):
            raise ValueError("Invalid desktop settings")
        return value
    return {}


def selected_precision(config: dict, total: int | None) -> str:
    return config.get("siglip_precision", recommended_precision(total))


def model_root(root: Path, config: dict) -> Path:
    if config.get("models"):
        return Path(config["models"]).expanduser().resolve()
    return models_directory(root) if metadata(root) else root / "models"


def bind(port: int) -> socket.socket:
    """Reserve loopback before opening a browser; never attach to an unknown server."""
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(128)
        return sock
    except BaseException:
        sock.close()
        raise


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import uvicorn
    from realopen_lens.api.app import create_app
    from realopen_lens.desktop.view import LauncherView
    from realopen_lens.runtime.readiness import inspect, CheckCancelled, OCR_GUIDE

    # Finder's PATH omits common local Tesseract installations.
    if sys.platform == "darwin":
        os.environ["PATH"] = os.pathsep.join(
            [os.environ.get("PATH", ""), "/opt/homebrew/bin", "/usr/local/bin"]
        )
    parser = argparse.ArgumentParser(description="RealOpen Lens desktop launcher")
    parser.add_argument("--workspace", type=Path, default=default_directory())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Use a valid port")
    url = f"http://127.0.0.1:{args.port}"
    root = args.workspace.expanduser().resolve()
    window = tk.Tk()
    window.title("RealOpen Lens")
    try:
        root.mkdir(parents=True, exist_ok=True)
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        handler = RotatingFileHandler(logs / "desktop.log", maxBytes=2_000_000, backupCount=3)
    except OSError as error:
        messagebox.showerror(
            "Workspace unavailable",
            f"Lens cannot write to {root}.\n\n"
            f"Restore folder permissions or choose a writable workspace.\n\n{error}",
        )
        window.destroy()
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    icon_path = Path(__file__).parent / "assets/Lens.png"
    if icon_path.is_file():
        window.iconphoto(True, tk.PhotoImage(file=str(icon_path)))
    status = tk.StringVar(value="Starting the local gallery…")
    try:
        config = settings(root)
        models = model_root(root, config)
    except (ValueError, OSError) as error:
        messagebox.showerror("Workspace settings", str(error))
        window.destroy()
        return
    total_memory = memory_bytes()
    initial_precision = selected_precision(config, total_memory)
    precision = tk.StringVar(value=initial_precision)
    languages = tk.StringVar(value=config.get("ocr_languages") or "eng+fra+ara")
    setup_status = tk.StringVar(value="Checking this device…")
    setup_details = tk.StringVar(
        value="Models, text recognition and workspace access are checked offline."
    )
    setup_progress = tk.DoubleVar(value=0)
    checker = None
    check_cancelled = threading.Event()
    check_pending = False
    check_revision = 0
    server = None
    worker = None
    reserved = None
    closing = False
    opened = False
    downloader = None
    cancel_download = threading.Event()
    events: queue.Queue[Any] = queue.Queue()

    def save():
        from realopen_lens.storage.artifacts import atomic_bytes

        if not re.fullmatch(r"(?:[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*)?", languages.get()):
            setup_status.set("Use OCR codes separated by +, for example eng+fra.")
            return False
        try:
            config["siglip_precision"] = precision.get()
            config["ocr_languages"] = languages.get()
            atomic_bytes(
                root / "desktop.json",
                json.dumps(
                    {
                        "models": str(models),
                        "theme": config.get("theme", "light"),
                        "siglip_precision": precision.get(),
                        "ocr_languages": languages.get(),
                    },
                    ensure_ascii=False,
                ).encode(),
            )
            return True
        except OSError as error:
            setup_status.set(f"Cannot save preferences: {error}")
            return False

    def request_check():
        nonlocal check_pending, check_revision
        if not re.fullmatch(r"(?:[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*)?", languages.get()):
            setup_status.set("Use OCR codes separated by +, for example eng+fra.")
            return
        if not save():
            return
        check_revision += 1
        check_pending = True
        check_cancelled.set()

    def launch_check():
        nonlocal checker, check_pending
        check_pending = False
        check_cancelled.clear()
        selected, requested, target, revision = (
            precision.get(),
            languages.get(),
            models,
            check_revision,
        )
        view.check_setup.configure(state="disabled")
        setup_status.set("Checking this device…")
        setup_progress.set(0)

        def progress(value):
            if check_cancelled.is_set():
                raise CheckCancelled()
            events.put({"kind": "setup_progress", "revision": revision, "value": value})

        def check():
            try:
                report = inspect(
                    root,
                    target,
                    siglip_precision=selected,
                    languages=requested,
                    on_progress=progress,
                )
                events.put({"kind": "setup_result", "revision": revision, "value": report})
            except CheckCancelled:
                pass
            except Exception as error:
                logging.exception("Device setup check failed")
                events.put({"kind": "setup_error", "revision": revision, "value": str(error)})

        checker = threading.Thread(target=check, daemon=True)
        checker.start()

    def choose_models():
        nonlocal models
        value = filedialog.askdirectory(title="Choose existing Lens model directory")
        if value:
            models = Path(value).resolve()
            save()
            view.model_label.configure(text=str(models))
            request_check()

    def download_models():
        nonlocal downloader
        from realopen_lens.models.manager import fetch

        if downloader and downloader.is_alive():
            cancel_download.set()
            return
        selected = precision.get()
        if not messagebox.askyesno(
            "Download models",
            f"Download SigLIP2 {selected.upper()} and E5 INT8 from Hugging Face? "
            "This requires internet access. Other installed model precisions are preserved.",
        ):
            return
        cancel_download.clear()
        check_cancelled.set()
        target = models

        def progress(name, received, total):
            if cancel_download.is_set():
                raise InterruptedError("Model download cancelled")
            events.put(f"{name}: {received / 1e6:.1f} / {total / 1e6:.1f} MB")

        def download():
            try:
                fetch(
                    target,
                    "int8",
                    events.put,
                    siglip_precision=selected,
                    download_progress=progress,
                )
                events.put(f"SigLIP2 {selected.upper()} and E5 INT8 verified and ready.")
            except Exception as error:
                logging.exception("Model setup stopped")
                events.put(f"Model setup: {error}")
            finally:
                events.put({"kind": "recheck"})

        downloader = threading.Thread(target=download, daemon=True)
        downloader.start()

    def start():
        nonlocal server, worker, reserved, opened
        if worker and worker.is_alive():
            if server and server.started:
                webbrowser.open(url)
            return
        try:
            reserved = bind(args.port)
            api = create_app(
                index_dir=root,
                models=models,
                port=args.port,
                preferred_siglip_precision=lambda: config.get(
                    "siglip_precision", initial_precision
                ),
                preferred_ocr_languages=lambda: config.get("ocr_languages", ""),
            )
            server = uvicorn.Server(
                uvicorn.Config(
                    api,
                    host="127.0.0.1",
                    port=args.port,
                    log_config=None,
                    access_log=False,
                    loop="asyncio",
                    http="h11",
                    ws="none",
                )
            )
        except (OSError, ValueError) as error:
            if reserved:
                reserved.close()
            status.set(f"Cannot start: {error}. Check whether Lens is already running.")
            return
        opened = False
        active_server, active_socket = server, reserved

        def serve():
            try:
                active_server.run(sockets=[active_socket])
            except BaseException:
                logging.exception("Desktop server failed")
                events.put("Server failed. See desktop.log for details.")
            finally:
                active_socket.close()

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        status.set("Starting local gallery…")
        view.choose_models.configure(state="disabled")

    def quit_app():
        nonlocal closing
        closing = True
        cancel_download.set()
        check_cancelled.set()
        status.set("Stopping local tasks and releasing models…")
        if server:
            server.should_exit = True

    def tick():
        nonlocal opened, worker, server
        while not events.empty():
            event = events.get_nowait()
            if isinstance(event, str):
                status.set(event)
            elif event["kind"] == "recheck":
                if not closing:
                    request_check()
            elif event.get("revision") == check_revision:
                value = event["value"]
                if event["kind"] == "setup_progress":
                    setup_status.set(value["phase"])
                    partial = min(
                        1, value.get("bytes_completed", 0) / max(1, value.get("total_bytes", 1))
                    )
                    setup_progress.set(100 * (value["step"] + partial) / value["max_steps"])
                elif event["kind"] == "setup_error":
                    view.set_setup_expanded(True)
                    setup_status.set(
                        f"Setup check stopped: {value}. Check again after resolving this issue."
                    )
                else:
                    setup_progress.set(100)
                    stale = value["siglip_precision"] != precision.get() or (
                        bool(languages.get()) and value["languages"] != languages.get()
                    )
                    if not stale and not value["ready"]:
                        view.set_setup_expanded(True)
                    setup_status.set(
                        "Settings changed. Check setup again."
                        if stale
                        else (
                            "Ready to add images. Open the library → Add folder."
                            if value["ready"]
                            else "Setup needs attention. You can still browse indexed images."
                        )
                    )
                    installed = value["ocr"].get("installed_languages", [])
                    preferred = [code for code in ("eng", "fra", "ara") if code in installed]
                    preview = preferred + [code for code in installed if code not in preferred]
                    setup_details.set(
                        "\n\n".join(
                            f"{'✓' if check['ok'] else '!'} {check['title']}: {check['detail']}"
                            + ("\n" + check.get("action", "") if not check["ok"] else "")
                            for check in value["checks"]
                        )
                        + f"\n\nInstalled OCR languages ({len(installed)}): "
                        + (", ".join(preview[:10]) or "None detected")
                        + (" …" if len(installed) > 10 else "")
                    )
                    if not languages.get() and value["languages"]:
                        languages.set(value["languages"])
                        save()
        if checker and not checker.is_alive():
            view.check_setup.configure(state="normal")
        if (
            not closing
            and check_pending
            and (not checker or not checker.is_alive())
            and (not downloader or not downloader.is_alive())
        ):
            launch_check()
        if closing:
            if (
                (not worker or not worker.is_alive())
                and (not downloader or not downloader.is_alive())
                and (not checker or not checker.is_alive())
            ):
                window.destroy()
                return
        elif worker and server:
            if server.started and worker.is_alive() and not opened:
                opened = True
                status.set(f"Gallery running locally at {url}")
                if not args.no_browser:
                    webbrowser.open(url)
            elif not worker.is_alive():
                status.set("Gallery stopped. See desktop.log if startup failed.")
                view.choose_models.configure(state="normal")
                worker, server = None, None
        window.after(200, tick)

    def theme_changed(theme):
        config["theme"] = theme
        save()

    def precision_changed():
        save()
        request_check()

    view = LauncherView(
        window,
        workspace=root,
        models=models,
        status=status,
        theme=config.get("theme", "light"),
        on_theme=theme_changed,
        precision=precision,
        total_memory=total_memory,
        recommendation=recommended_precision(total_memory),
        on_precision=precision_changed,
        languages=languages,
        setup_status=setup_status,
        setup_details=setup_details,
        setup_progress=setup_progress,
        actions={
            "open": start,
            "stop": lambda: setattr(server, "should_exit", True) if server else None,
            "models": choose_models,
            "download": download_models,
            "check": request_check,
            "ocr_changed": save,
            "ocr_help": lambda: webbrowser.open(OCR_GUIDE),
            "logs": lambda: webbrowser.open(logs.as_uri()),
            "quit": quit_app,
        },
    )
    window.protocol("WM_DELETE_WINDOW", quit_app)
    window.after(200, tick)
    window.after(400, start)
    window.after(500, request_check)
    window.mainloop()


if __name__ == "__main__":
    main()
