"""Desktop lifecycle controller with no shell commands or implicit downloads."""

import json
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import shutil
import socket
import sys
import threading
import webbrowser

import psutil

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


def memory_bytes() -> int | None:
    try:
        return psutil.virtual_memory().total
    except (OSError, NotImplementedError, psutil.Error):
        return None


def recommended_precision(total: int | None) -> str:
    """Use installed physical RAM, not momentary free memory; fall back conservatively."""
    # Decimal GB keeps the 16 GB class eligible even when the OS reserves a little
    # of a nominal 16 GiB installation for hardware.
    return "fp32" if total is not None and total >= 16_000_000_000 else "int8"


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
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    handler = RotatingFileHandler(logs / "desktop.log", maxBytes=2_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    window = tk.Tk()
    window.title("RealOpen Lens")
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
    server = None
    worker = None
    reserved = None
    closing = False
    opened = False
    downloader = None
    cancel_download = threading.Event()
    events: queue.Queue[str] = queue.Queue()

    def save():
        from realopen_lens.storage.artifacts import atomic_bytes

        config["siglip_precision"] = precision.get()
        atomic_bytes(
            root / "desktop.json",
            json.dumps(
                {
                    "models": str(models),
                    "theme": config.get("theme", "light"),
                    "siglip_precision": precision.get(),
                },
                ensure_ascii=False,
            ).encode(),
        )

    def choose_models():
        nonlocal models
        value = filedialog.askdirectory(title="Choose existing Lens model directory")
        if value:
            models = Path(value).resolve()
            save()
            view.model_label.configure(text=str(models))

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

        def serve():
            try:
                server.run(sockets=[reserved])
            except BaseException:
                logging.exception("Desktop server failed")
                events.put("Server failed. See desktop.log for details.")
            finally:
                reserved.close()

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        status.set("Starting local gallery…")
        view.choose_models.configure(state="disabled")

    def quit_app():
        nonlocal closing
        closing = True
        cancel_download.set()
        status.set("Stopping local tasks and releasing models…")
        if server:
            server.should_exit = True

    def tick():
        nonlocal opened
        while not events.empty():
            status.set(events.get_nowait())
        if closing:
            if (not worker or not worker.is_alive()) and (
                not downloader or not downloader.is_alive()
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
        window.after(200, tick)

    def theme_changed(theme):
        config["theme"] = theme
        save()

    def precision_changed():
        save()

    view = LauncherView(
        window,
        workspace=root,
        models=models,
        status=status,
        ocr=shutil.which("tesseract"),
        theme=config.get("theme", "light"),
        on_theme=theme_changed,
        precision=precision,
        total_memory=total_memory,
        recommendation=recommended_precision(total_memory),
        on_precision=precision_changed,
        actions={
            "open": start,
            "stop": lambda: setattr(server, "should_exit", True) if server else None,
            "models": choose_models,
            "download": download_models,
            "logs": lambda: webbrowser.open(logs.as_uri()),
            "quit": quit_app,
        },
    )
    window.protocol("WM_DELETE_WINDOW", quit_app)
    window.after(200, tick)
    window.after(400, start)
    window.mainloop()


if __name__ == "__main__":
    main()
