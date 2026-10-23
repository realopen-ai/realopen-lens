"""Linux archive boundaries, OCR linkage and candidate acceptance regressions."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def scripts(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "packaging"
    monkeypatch.syspath_prepend(str(root))

    def load(name):
        spec = importlib.util.spec_from_file_location(f"test_{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def tarball(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, data, link in entries:
            member = tarfile.TarInfo(name)
            if link is not None:
                member.type, member.linkname = tarfile.SYMTYPE, link
                archive.addfile(member)
            else:
                member.mode, member.size = 0o755, len(data)
                archive.addfile(member, io.BytesIO(data))


def test_linux_archive_preserves_executable_and_confined_symlinks(tmp_path, scripts):
    archive = tmp_path / "app.tar.gz"
    tarball(
        archive,
        [
            ("RealOpenLens/RealOpenLens", b"ELF", None),
            ("RealOpenLens/_internal/library.so", b"library", None),
            ("RealOpenLens/lib.so", b"", "_internal/library.so"),
        ],
    )
    target = tmp_path / "out"
    scripts("linux_archive").extract(archive, target)
    executable = target / "RealOpenLens/RealOpenLens"
    assert executable.stat().st_mode & 0o111
    assert (target / "RealOpenLens/lib.so").read_bytes() == b"library"


@pytest.mark.parametrize(
    "name,link",
    [
        ("../outside", None),
        ("/absolute", None),
        ("other-app/file", None),
        ("RealOpenLens/link", "../../outside"),
        ("RealOpenLens/link", "/tmp/outside"),
    ],
)
def test_linux_archive_rejects_escape(tmp_path, scripts, name, link):
    archive = tmp_path / "app.tar.gz"
    tarball(archive, [(name, b"file", link)])
    with pytest.raises(ValueError, match="escapes"):
        scripts("linux_archive").extract(archive, tmp_path / "out")
    assert not (tmp_path / "outside").exists()


def test_linux_archive_rejects_links_redirected_by_another_member(tmp_path, scripts):
    archive = tmp_path / "app.tar.gz"
    tarball(
        archive,
        [("RealOpenLens/redirect", b"", ".."), ("RealOpenLens/redirect/outside", b"bad", None)],
    )
    with pytest.raises(ValueError, match="escapes"):
        scripts("linux_archive").extract(archive, tmp_path / "out")


def test_linux_ocr_elf_architecture_is_checked(tmp_path, scripts):
    ocr = scripts("bundle_linux_ocr")
    binary = tmp_path / "engine"
    header = bytearray(20)
    header[:6], header[18:20] = b"\x7fELF\x02\x01", b"\x3e\x00"
    binary.write_bytes(header)
    ocr.verify_elf(binary)
    header[18:20] = b"\xb7\x00"  # AArch64
    binary.write_bytes(header)
    with pytest.raises(ValueError, match="x86-64"):
        ocr.verify_elf(binary)


def test_linux_ocr_transitive_libraries_exclude_only_glibc(tmp_path, scripts, monkeypatch):
    ocr = scripts("bundle_linux_ocr")
    monkeypatch.setattr(
        ocr,
        "run",
        lambda *a: """
 linux-vdso.so.1 (0x123)
 libtesseract.so.5 => /lib/libtesseract.so.5 (0x123)
 libstdc++.so.6 => /lib/libstdc++.so.6 (0x123)
 libgcc_s.so.1 => /lib/libgcc_s.so.1 (0x123)
 libc.so.6 => /lib/libc.so.6 (0x123)
 /lib64/ld-linux-x86-64.so.2 (0x123)
""",
    )
    libraries = ocr.dependencies(tmp_path / "tesseract")
    assert set(libraries) == {"libtesseract.so.5", "libstdc++.so.6", "libgcc_s.so.1"}
    monkeypatch.setattr(ocr, "run", lambda *a: "libmissing.so => not found")
    with pytest.raises(ValueError, match="Unresolved"):
        ocr.dependencies(tmp_path / "tesseract")


def test_linux_model_acceptance_extracts_executable(tmp_path, scripts, monkeypatch):
    acceptance = scripts("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", "linux")
    monkeypatch.setattr(acceptance.platform, "platform", lambda: "test-linux")
    archive = tmp_path / "candidate.tar.gz"
    tarball(archive, [("RealOpenLens/RealOpenLens", b"fake ELF", None)])

    def run(command, **kwargs):
        if Path(command[1]).name == "smoke.py":
            executable = Path(command[2])
            assert executable.name == "RealOpenLens" and executable.is_file()
            assert executable.stat().st_mode & 0o111
            kwargs["stdout"].write(json.dumps({"siglip_precision": command[-1]}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    report = acceptance.validate(archive, tmp_path / "evidence", models=tmp_path / "models")
    assert report["inference"] == "passed" and set(report["results"]) == {"int8", "fp32"}
    assert report["notarization"] == "not_applicable"


def test_linux_rejects_notarization_before_execution(tmp_path, scripts, monkeypatch):
    acceptance = scripts("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", "linux")
    with pytest.raises(ValueError, match="only to macOS"):
        acceptance.validate(tmp_path / "app.tar.gz", tmp_path / "out", require_notarized=True)
    assert not (tmp_path / "out").exists()
