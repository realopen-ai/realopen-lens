"""Bundle a pinned, statically linked vcpkg OCR build and its full notices."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from bundle_ocr import checksum, language_data

# Windows-provided libraries that the pinned OCR dependency graph may use.
# Presence in System32 alone is insufficient: third-party installers can put
# their own DLLs there on a build machine.
SYSTEM_DLLS = frozenset(
    "advapi32 bcrypt comdlg32 crypt32 gdi32 iphlpapi kernel32 kernelbase msvcrt "
    "netapi32 normaliz ntdll ole32 oleaut32 rpcrt4 secur32 setupapi shell32 "
    "shlwapi ucrtbase user32 userenv version winhttp wininet winmm ws2_32".split()
)


def verify_pe(executable: Path):
    """Reject the wrong architecture and any non-system DLL requirement."""
    import pefile

    with pefile.PE(str(executable)) as binary:
        if binary.FILE_HEADER.Machine != 0x8664:
            raise ValueError("Windows OCR must be an x64 PE executable")
        imports = {
            item.dll.decode("ascii").lower()
            for table in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT")
            for item in getattr(binary, table, [])
        }
    system = Path(os.environ["SystemRoot"]) / "System32"
    # Static /MT builds should have only Windows-provided dependencies. In
    # particular, never accept a VC runtime merely because the builder has it.
    for name in sorted(imports):
        contract = name.startswith(("api-ms-win-", "ext-ms-win-"))
        os_library = name.endswith(".dll") and name[:-4] in SYSTEM_DLLS
        if not contract and not (os_library and (system / name).is_file()):
            raise ValueError(f"OCR is not self-contained: {name}")
    return sorted(imports)


def bundle_ocr(bundle: Path, installed: Path, source_data: Path | None = None):
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "packaging/tessdata.json").read_text(encoding="utf-8"))
    executable = installed / "tools/tesseract/tesseract.exe"
    imports = verify_pe(executable)
    result = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
        env={**os.environ, "PATH": str(Path(os.environ["SystemRoot"]) / "System32")},
    )
    version = result.stdout.splitlines()[0]
    if version != "tesseract 5.5.3":
        raise ValueError(f"Unexpected pinned OCR version: {version}")
    resources = bundle / "_internal/lens-ocr"
    engine = resources / "bin/tesseract.exe"
    engine.parent.mkdir(parents=True)
    shutil.copyfile(executable, engine)
    language_data(resources / "tessdata", source_data or root / "build/tessdata", registry)

    notices = sorted(installed.glob("share/*/copyright"))
    if not notices or not any(path.parent.name == "tesseract" for path in notices):
        raise ValueError("Missing vcpkg OCR copyright notices")
    notice_dir = resources / "licenses"
    notice_dir.mkdir()
    records = [{"file": engine.relative_to(bundle).as_posix(), "sha256": checksum(engine)}]
    for notice in notices:
        target = notice_dir / f"{notice.parent.name}.txt"
        shutil.copyfile(notice, target)
        records.append({"file": target.relative_to(bundle).as_posix(), "sha256": checksum(target)})
    status = installed.parent / "vcpkg/status"
    shutil.copyfile(status, resources / "vcpkg-status.txt")
    records.append(
        {
            "file": (resources / "vcpkg-status.txt").relative_to(bundle).as_posix(),
            "sha256": checksum(resources / "vcpkg-status.txt"),
        }
    )
    manifest = {
        "version": 1,
        "engine": version,
        "build": json.loads((root / "packaging/windows-ocr/vcpkg.json").read_text("utf-8")),
        "imports": imports,
        "files": records,
        "tessdata": registry,
        "tsv_sha256": checksum(resources / "tessdata/configs/tsv"),
    }
    (resources / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    public = bundle / "_internal/licenses/realopen-lens/THIRD_PARTY_NOTICES.md"
    with public.open("a", encoding="utf-8") as stream:
        stream.write("\n## Windows OCR dependency notices\n\n")
        for notice in notices:
            stream.write(f"\n### {notice.parent.name}\n\n~~~~~~~~text\n")
            stream.write(notice.read_text(encoding="utf-8"))
            stream.write("\n~~~~~~~~\n")
    return manifest
