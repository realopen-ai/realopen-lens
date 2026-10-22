"""Offline, bounded setup checks shared by the desktop and gallery adapters."""

from collections.abc import Callable
import hashlib
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile

from realopen_lens.models.manager import artifacts, artifact_path
from realopen_lens.storage.catalog import Catalog
from realopen_lens.ocr import runtime as ocr_runtime

OCR_GUIDE = "https://tesseract-ocr.github.io/tessdoc/Installation.html"


class CheckCancelled(KeyboardInterrupt):
    pass


def installed_ocr() -> dict:
    executable = ocr_runtime.resolve(shutil.which("tesseract"))
    if not executable:
        raise ValueError("Tesseract 5 is not installed or cannot be found")
    ocr_runtime.verify(executable)

    def run(argument: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [executable, argument],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env=ocr_runtime.environment(executable),
        )

    version = run("--version")
    match = re.search(r"tesseract\s+(\d+)\S*", version.stdout, re.IGNORECASE)
    if not match or int(match[1]) < 5:
        raise ValueError("Tesseract 5 or newer is required")
    listed = run("--list-langs")
    languages = sorted(
        {
            line.strip()
            for line in listed.stdout.splitlines()
            if re.fullmatch(r"[A-Za-z0-9_]+", line.strip()) and line.strip() not in {"osd", "equ"}
        }
    )
    return {
        "executable": executable,
        "version": version.stdout.splitlines()[0],
        "installed_languages": languages,
    }


def suggested_languages(installed: list[str]) -> str:
    preferred = [code for code in ("eng", "fra", "ara") if code in installed]
    return "+".join(preferred or installed[:1])


def inspect(
    workspace: Path,
    models: Path,
    *,
    siglip_precision: str = "int8",
    languages: str = "",
    source: Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Verify files without loading inference sessions, scanning images or using a network."""
    selected = artifacts("int8", siglip_precision)
    if languages and not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", languages):
        raise ValueError("Use installed OCR language codes separated by +, for example eng+fra")
    checks = []

    def progress(phase: str, step: int, **values):
        if on_progress:
            on_progress({"phase": phase, "step": step, "max_steps": 4 if source else 3, **values})

    def record(key, title, action, operation):
        try:
            detail = operation()
            checks.append({"id": key, "title": title, "ok": True, "detail": detail})
        except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
            checks.append(
                {"id": key, "title": title, "ok": False, "detail": str(error), "action": action}
            )

    def writable():
        workspace.mkdir(parents=True, exist_ok=True)
        from realopen_lens.workspace.identity import metadata

        metadata(workspace)
        with tempfile.TemporaryFile(dir=workspace) as probe:
            probe.write(b"lens setup check")
            probe.flush()
        if (workspace / "catalog.sqlite3").exists():
            catalog = Catalog(workspace, readonly=True)
            catalog.close()
        return str(workspace)

    progress("Checking workspace access", 0)
    record(
        "workspace",
        "Workspace",
        "Choose a writable workspace or restore folder permissions.",
        writable,
    )
    progress("Verifying local model files", 1)
    total = sum(spec["size"] for _, _, spec in selected)
    completed = 0
    issues = []
    for name, key, spec in selected:
        path = artifact_path(models, name, key)
        label = f"{name}/{spec['path']}"
        try:
            if not path.is_file() or path.stat().st_size != spec["size"]:
                issues.append(f"Missing or incomplete: {label}")
            else:
                checksum = hashlib.sha256()
                with path.open("rb") as stream:
                    while block := stream.read(4 * 1024 * 1024):
                        checksum.update(block)
                        progress(
                            "Verifying local model files",
                            1,
                            current_file=label,
                            bytes_completed=completed + stream.tell(),
                            total_bytes=total,
                        )
                if checksum.hexdigest() != spec["sha256"]:
                    issues.append(f"Integrity check failed: {label}")
        except OSError as error:
            issues.append(f"Cannot read {label}: {error}")
        completed += spec["size"]
    checks.append(
        {
            "id": "models",
            "title": f"SigLIP2 {siglip_precision.upper()} + E5 INT8",
            "ok": not issues,
            "detail": "; ".join(issues) if issues else "All pinned files verified",
            "action": "Choose the existing model folder, or use Download models "
            "in the desktop launcher.",
        }
    )
    progress("Checking Tesseract and language packs", 2)
    ocr = {}

    def check_ocr():
        nonlocal ocr, languages
        ocr = installed_ocr()
        languages = languages or suggested_languages(ocr["installed_languages"])
        missing = set(languages.split("+")) - set(ocr["installed_languages"])
        if not languages:
            raise ValueError("No OCR language packs are installed")
        if missing:
            raise ValueError("Missing OCR language packs: " + ", ".join(sorted(missing - {""})))
        return f"{ocr['version']} · {languages}"

    record(
        "ocr",
        "Text recognition",
        (
            "Select the bundled eng/fra/ara languages. If bundled OCR is damaged, reinstall Lens."
            if ocr_runtime.bundle_paths()
            else "Install Tesseract 5 and the requested language packs, "
            "or select installed languages; then check again."
        ),
        check_ocr,
    )
    if source is not None:
        progress("Checking source folder access", 3)

        def readable():
            if not source.is_dir():
                raise ValueError("Source folder is unavailable")
            with os.scandir(source) as entries:
                next(entries, None)
            return str(source)

        record(
            "source",
            "Image folder",
            "Reconnect this folder or grant Lens access "
            "in your operating system's privacy settings.",
            readable,
        )
    progress("Setup check finished", 4 if source else 3)
    return {
        "ready": all(check["ok"] for check in checks),
        "checks": checks,
        "siglip_precision": siglip_precision,
        "languages": languages,
        "source": str(source) if source else None,
        "ocr": ocr,
        "ocr_guide": OCR_GUIDE,
    }
