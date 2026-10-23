"""Portable Windows OCR boundaries, tested without executing Windows binaries."""

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from realopen_lens.ocr import runtime


def script(name, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "packaging"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_frozen_windows_uses_private_runtime_without_mac_layout(tmp_path, monkeypatch):
    executable = tmp_path / "Portable Lens/RealOpenLens.exe"
    with monkeypatch.context() as patch:
        patch.setattr(runtime.sys, "frozen", True, raising=False)
        patch.setattr(runtime.sys, "platform", "win32")
        patch.setattr(runtime.sys, "executable", str(executable))
        found = runtime.bundle_paths()
    private = executable.parent / "_internal/lens-ocr"
    assert found == (private / "bin/tesseract.exe", private)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_source_runtime_does_not_require_private_ocr(monkeypatch, platform):
    monkeypatch.setattr(runtime.sys, "frozen", False, raising=False)
    monkeypatch.setattr(runtime.sys, "platform", platform)
    assert runtime.bundle_paths() is None


@pytest.mark.parametrize("architecture", [0x14C, 0xAA64])
def test_windows_ocr_rejects_non_x64_before_copying(tmp_path, monkeypatch, architecture):
    module = script("bundle_windows_ocr", monkeypatch)
    binary = SimpleNamespace(FILE_HEADER=SimpleNamespace(Machine=architecture))
    monkeypatch.setitem(sys.modules, "pefile", SimpleNamespace(PE=lambda *a: nullcontext(binary)))
    with pytest.raises(ValueError, match="x64"):
        module.verify_pe(tmp_path / "tesseract.exe")


@pytest.mark.parametrize("dll", ["libomp140.dll", "vcruntime140.dll", "vcomp140.dll", "other.dll"])
def test_windows_ocr_rejects_builder_only_dlls(tmp_path, monkeypatch, dll):
    module = script("bundle_windows_ocr", monkeypatch)
    system = tmp_path / "System32"
    system.mkdir()
    # System32 can contain third-party DLLs as well as an installed VC runtime.
    # Finding a file there must not make it a valid Windows system dependency.
    (system / dll).touch()
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    binary = SimpleNamespace(
        FILE_HEADER=SimpleNamespace(Machine=0x8664),
        DIRECTORY_ENTRY_DELAY_IMPORT=[SimpleNamespace(dll=dll.encode())],
    )
    monkeypatch.setitem(sys.modules, "pefile", SimpleNamespace(PE=lambda *a: nullcontext(binary)))
    with pytest.raises(ValueError, match="not self-contained"):
        module.verify_pe(tmp_path / "tesseract.exe")


def test_windows_ocr_accepts_windows_libraries_and_api_contracts(tmp_path, monkeypatch):
    module = script("bundle_windows_ocr", monkeypatch)
    system = tmp_path / "System32"
    system.mkdir()
    (system / "kernel32.dll").touch()
    (system / "xmllite.dll").touch()
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    names = ["KERNEL32.dll", "XmlLite.dll", "api-ms-win-core-file-l1-1-0.dll"]
    binary = SimpleNamespace(
        FILE_HEADER=SimpleNamespace(Machine=0x8664),
        DIRECTORY_ENTRY_IMPORT=[SimpleNamespace(dll=name.encode()) for name in names],
    )
    monkeypatch.setitem(sys.modules, "pefile", SimpleNamespace(PE=lambda *a: nullcontext(binary)))
    assert module.verify_pe(tmp_path / "tesseract.exe") == sorted(name.lower() for name in names)


def test_windows_bundle_manifest_notices_and_runtime_are_compatible(tmp_path, monkeypatch):
    module = script("bundle_windows_ocr", monkeypatch)
    installed = tmp_path / "installed/lens-x64-windows-static"
    engine = installed / "tools/tesseract/tesseract.exe"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"fake x64 executable, never executed")
    for name in ("tesseract", "leptonica", "zlib"):
        notice = installed / "share" / name / "copyright"
        notice.parent.mkdir(parents=True)
        notice.write_text(f"License for {name}", encoding="utf-8")
    (installed.parent / "vcpkg").mkdir()
    (installed.parent / "vcpkg/status").write_text("Package: tesseract\nVersion: 5.5.3\n")
    bundle = tmp_path / "Portable Lens"
    public = bundle / "_internal/licenses/realopen-lens/THIRD_PARTY_NOTICES.md"
    public.parent.mkdir(parents=True)
    public.write_text("# Third-party notices\n", encoding="utf-8")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    monkeypatch.setattr(module, "verify_pe", lambda p: ["kernel32.dll"])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="tesseract 5.5.3\n", returncode=0),
    )

    def languages(destination, source, registry):
        (destination / "configs").mkdir(parents=True)
        (destination / "configs/tsv").write_bytes(b"tsv\n")
        for name, spec in registry.items():
            data = name.encode()
            (destination / f"{name}.traineddata").write_bytes(data)
            spec["sha256"] = hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(module, "language_data", languages)
    manifest = module.bundle_ocr(bundle, installed)
    resources = bundle / "_internal/lens-ocr"
    engine = resources / "bin/tesseract.exe"
    monkeypatch.setattr(runtime, "bundle_paths", lambda: (engine, resources))
    assert runtime.verify(str(engine)) == manifest
    assert set(manifest["tessdata"]) == {"eng", "fra", "ara", "osd"}
    assert "License for leptonica" in public.read_text(encoding="utf-8")
    # Dependency notices are integrity protected along with the executable/data.
    (resources / "licenses/zlib.txt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity check failed"):
        runtime.verify(str(engine))


def test_windows_build_rejects_wrong_baseline_before_native_build(tmp_path, monkeypatch):
    module = script("build_windows", monkeypatch)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="wrong-revision\n")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(module.sys, "argv", ["build_windows", "--vcpkg-root", str(tmp_path)])
    with monkeypatch.context() as patch:
        patch.setattr(module.sys, "platform", "win32")
        with pytest.raises(SystemExit, match="Use vcpkg revision"):
            module.main()
    assert len(commands) == 1 and "rev-parse" in commands[0]


def test_windows_extraction_rejects_traversal_and_preserves_failure_report(tmp_path, monkeypatch):
    import zipfile

    module = script("ci_smoke", monkeypatch)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../outside.txt", "escape")
    report = tmp_path / "report.json"
    with monkeypatch.context() as patch:
        patch.setattr(module.sys, "platform", "win32")
        with pytest.raises(ValueError, match="escapes"):
            module.validate(archive, report)
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == result["checks"][0]["status"] == "failed"
    assert result["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert not (tmp_path / "outside.txt").exists()


def test_windows_build_cannot_cross_compile(tmp_path):
    if sys.platform == "win32":
        pytest.skip("Non-Windows build guard")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "packaging/build_windows.py"), "--vcpkg-root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode and "Build Windows x64 artifacts on Windows x64" in result.stderr


@pytest.mark.parametrize("failure", [None, "attach", "signal"])
def test_windows_shutdown_helper_targets_only_owned_group(monkeypatch, failure):
    module = script("windows_signal", monkeypatch)
    calls = []
    kernel = SimpleNamespace(
        FreeConsole=lambda: calls.append("detach"),
        AttachConsole=lambda pid: calls.append(("attach", pid)) or failure != "attach",
        GenerateConsoleCtrlEvent=lambda event, pid: calls.append(("signal", event, pid))
        or failure != "signal",
    )
    monkeypatch.setattr(module.ctypes, "WinDLL", lambda *a, **kw: kernel, raising=False)
    monkeypatch.setattr(module.ctypes, "get_last_error", lambda: 5, raising=False)
    with monkeypatch.context() as patch:
        patch.setattr(module.sys, "platform", "win32")
        if failure:
            with pytest.raises(OSError):
                module.send(1234)
        else:
            module.send(1234)
    assert calls[:2] == ["detach", ("attach", 1234)]
    if failure == "attach":
        assert len(calls) == 2
    else:
        assert calls[2:] == [("signal", 1, 1234), "detach"]
