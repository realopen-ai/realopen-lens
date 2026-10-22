"""Offline packaging diagnostics: no model weights, GUI windows or application-data writes."""

from importlib import import_module
from importlib.metadata import version
import os
from pathlib import Path
import sys


def inspect_bundle() -> dict:
    if not getattr(sys, "frozen", False):
        raise ValueError("Bundle diagnostics must run from the frozen executable")
    from realopen_lens.runtime.offline import enforce

    enforce()
    packages = {}
    for module, distribution in (
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("onnxruntime", "onnxruntime"),
        ("tokenizers", "tokenizers"),
        ("psutil", "psutil"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("realopen_lens", "realopen-lens"),
    ):
        import_module(module)
        packages[distribution] = version(distribution)

    import tkinter
    from realopen_lens.models.manager import registry
    from realopen_lens.workspace.identity import build

    tcl = tkinter.Tcl()
    if not (Path(tcl.eval("info library")) / "init.tcl").is_file():
        raise ValueError("Missing bundled Tcl resources")
    tk_library = os.environ.get("TK_LIBRARY")
    if not tk_library or not (Path(tk_library) / "tk.tcl").is_file():
        raise ValueError("Missing bundled Tk resources")
    root = Path(__file__).resolve().parents[1]
    for resource in (
        "api/static/index.html",
        "api/static/assets/lens.svg",
        "desktop/assets/Lens.png",
    ):
        if not (root / resource).is_file():
            raise ValueError(f"Missing bundled resource: {resource}")
    if set(registry()["models"]) != {"siglip2", "e5"}:
        raise ValueError("Missing pinned model registry")
    from realopen_lens.runtime.readiness import installed_ocr

    ocr = installed_ocr()
    return {
        "frozen": True,
        "python": sys.version.split()[0],
        "packages": packages,
        "tcl": tcl.eval("info patchlevel"),
        "tk_resources": True,
        "build": build(),
        "models_downloaded": False,
        "ocr": ocr,
    }
