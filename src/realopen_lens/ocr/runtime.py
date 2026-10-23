"""Locate the desktop's private OCR runtime without changing host installations."""

import hashlib
import json
import os
from pathlib import Path
import sys


def bundle_paths() -> tuple[Path, Path] | None:
    if not getattr(sys, "frozen", False):
        return None
    if sys.platform in {"win32", "linux"}:
        private = Path(sys.executable).resolve().parent / "_internal/lens-ocr"
        return (
            private / ("bin/tesseract.exe" if sys.platform == "win32" else "bin/tesseract"),
            private,
        )
    if sys.platform != "darwin":
        return None
    contents = Path(sys.executable).resolve().parents[1]
    return (
        contents / "Frameworks/lens-ocr/bin/tesseract",
        contents / "Resources/lens-ocr",
    )


def resolve(system_executable: str | None) -> str | None:
    paths = bundle_paths()
    if paths is None:
        return system_executable
    executable, resources = paths
    if not executable.is_file() or not (resources / "runtime.json").is_file():
        raise ValueError("Bundled OCR is incomplete; reinstall the Lens application")
    return str(executable)


def environment(executable: str) -> dict[str, str]:
    result = dict(os.environ)
    paths = bundle_paths()
    if paths and Path(executable) == paths[0]:
        result["TESSDATA_PREFIX"] = str(paths[1] / "tessdata")
        # Frozen Python's loader overrides must not replace OCR's private libraries.
        for name in (
            "DYLD_LIBRARY_PATH",
            "DYLD_FALLBACK_LIBRARY_PATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
        ):
            result.pop(name, None)
    return result


def fingerprint(executable: str) -> str | None:
    paths = bundle_paths()
    if paths and Path(executable) == paths[0]:
        manifest = json.loads((paths[1] / "runtime.json").read_text(encoding="utf-8"))
        # Code signing, relocation, paths and notice files do not change OCR.
        # Original binary hashes still change for real engine/library updates.
        binaries = [
            (Path(row["file"]).name, row.get("source_sha256", row["sha256"]))
            for row in manifest["files"]
            if "/bin/" in row["file"] or "/lib/" in row["file"]
        ]
        value = {
            "engine": manifest["engine"],
            "binaries": sorted(binaries),
            "tsv_sha256": manifest["tsv_sha256"],
        }
        if not binaries:
            raise ValueError("Bundled OCR manifest contains no runtime binaries")
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return None


def verify(executable: str) -> dict | None:
    paths = bundle_paths()
    if not paths or Path(executable) != paths[0]:
        return None
    # Portable manifests use the directory root; macOS uses the .app root.
    portable = paths[1].parent.name == "_internal"
    contents = paths[1].parent if portable else paths[1].parents[1]
    manifest = json.loads((paths[1] / "runtime.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or set(manifest.get("tessdata", {})) != {"eng", "fra", "ara", "osd"}
        or not isinstance(manifest.get("files"), list)
        or not manifest["files"]
    ):
        raise ValueError("Invalid bundled OCR manifest; reinstall Lens")
    if not any(
        record.get("file") == paths[0].relative_to(contents.parent).as_posix()
        for record in manifest["files"]
    ):
        raise ValueError("Bundled OCR manifest does not identify its executable")
    records = [(contents.parent / record["file"], record["sha256"]) for record in manifest["files"]]
    records += [
        (paths[1] / "tessdata" / f"{name}.traineddata", spec["sha256"])
        for name, spec in manifest["tessdata"].items()
    ]
    records += [(paths[1] / "tessdata/configs/tsv", manifest["tsv_sha256"])]
    for path, expected in records:
        if not path.resolve().is_relative_to(contents.resolve()):
            raise ValueError("Bundled OCR resource escapes the application")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Bundled OCR integrity check failed: {path.name}; reinstall Lens")
    return manifest
