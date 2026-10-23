"""Bundle and relocate Ubuntu's OCR dependency closure with package notices."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from bundle_ocr import checksum, language_data

# These belong to the target glibc installation, never to a relocated application.
SYSTEM_LIBRARIES = frozenset(
    "libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libresolv.so.2 "
    "libutil.so.1 ld-linux-x86-64.so.2".split()
)


def run(*command):
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", check=True, timeout=60
    ).stdout


def verify_elf(path):
    with path.open("rb") as stream:
        header = stream.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\x3e\x00":
        raise ValueError(f"Linux OCR must be an x86-64 little-endian ELF: {path}")


def dependencies(executable):
    """ldd reports the entire transitive closure on the native build host."""
    libraries = {}
    for line in run("ldd", str(executable)).splitlines():
        if not line.strip():
            continue
        if "not found" in line:
            raise ValueError(f"Unresolved OCR dependency: {line.strip()}")
        match = re.match(r"\s*(\S+) => (/\S+) \(", line)
        if match:
            name, source = match.groups()
            if name not in SYSTEM_LIBRARIES:
                libraries[name] = Path(source)
        elif "linux-vdso" not in line and "ld-linux-x86-64.so.2" not in line:
            raise ValueError(f"Unexpected OCR dependency entry: {line.strip()}")
    if not libraries:
        raise ValueError("No dynamic OCR libraries discovered")
    return libraries


def package_for(path):
    # Ubuntu's merged-/usr layout can differ from the path in dpkg's file list.
    candidates = [path, path.resolve()]
    if str(path).startswith("/usr/"):
        candidates.append(Path(str(path)[4:]))
    elif str(path).startswith("/lib/"):
        candidates.append(Path("/usr" + str(path)))
    for candidate in candidates:
        result = subprocess.run(
            ["dpkg-query", "-S", str(candidate)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        for line in result.stdout.splitlines() if result.returncode == 0 else ():
            if ": " in line:
                return line.split(": ", 1)[0]
    raise ValueError(f"Missing Debian package provenance: {path}")


def bundle_ocr(bundle: Path, executable: Path, source_data: Path):
    verify_elf(executable)
    version = run(str(executable), "--version").splitlines()[0]
    if not re.fullmatch(r"tesseract 5\.\S+", version):
        raise ValueError(f"Build with Tesseract 5, found {version}")
    libraries = dependencies(executable)
    resources = bundle / "_internal/lens-ocr"
    (resources / "bin").mkdir(parents=True)
    (resources / "lib").mkdir()
    (resources / "licenses").mkdir()
    records, packages = [], {}
    for name, source in {"tesseract": executable, **libraries}.items():
        verify_elf(source)
        package = package_for(source)
        if package not in packages:
            packages[package] = run("dpkg-query", "-W", "-f=${Version}", package).strip()
            notice = Path("/usr/share/doc") / package.split(":", 1)[0] / "copyright"
            if not notice.is_file():
                raise ValueError(f"Missing copyright notice for {package}")
            target = resources / "licenses" / f"{package.replace(':', '_')}.txt"
            shutil.copyfile(notice, target)
            records.append(
                {"file": target.relative_to(bundle).as_posix(), "sha256": checksum(target)}
            )
        target = resources / ("bin" if name == "tesseract" else "lib") / name
        shutil.copyfile(source, target)
        target.chmod(0o755)
        run(
            "patchelf",
            "--force-rpath",
            "--set-rpath",
            "$ORIGIN/../lib" if name == "tesseract" else "$ORIGIN",
            str(target),
        )
        records.append(
            {
                "file": target.relative_to(bundle).as_posix(),
                "sha256": checksum(target),
                "source_sha256": checksum(source),
            }
        )
    # Debian copyright files may incorporate these license texts by reference.
    for notice in sorted(Path("/usr/share/common-licenses").iterdir()):
        if notice.is_file():
            target = resources / "licenses" / f"common-{notice.name}.txt"
            shutil.copyfile(notice, target)
            records.append(
                {"file": target.relative_to(bundle).as_posix(), "sha256": checksum(target)}
            )
    registry = json.loads(Path(__file__).with_name("tessdata.json").read_text(encoding="utf-8"))
    language_data(resources / "tessdata", source_data, registry)
    manifest = {
        "version": 1,
        "engine": version,
        "files": records,
        "packages": packages,
        "system_libraries": sorted(SYSTEM_LIBRARIES),
        "tessdata": registry,
        "tsv_sha256": checksum(resources / "tessdata/configs/tsv"),
    }
    (resources / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    public = bundle / "_internal/licenses/realopen-lens/THIRD_PARTY_NOTICES.md"
    with public.open("a", encoding="utf-8") as stream:
        stream.write("\n## Linux OCR dependency notices\n\n")
        for notice in sorted((resources / "licenses").glob("*.txt")):
            stream.write(f"\n### {notice.stem}\n\n~~~~~~~~text\n")
            stream.write(notice.read_text(encoding="utf-8"))
            stream.write("\n~~~~~~~~\n")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("LD_")}
    environment.update(PATH="", TESSDATA_PREFIX=str(resources / "tessdata"))
    copied = resources / "bin/tesseract"
    listed = subprocess.run(
        [str(copied), "--list-langs"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    if not set(registry).issubset(set(listed.splitlines()[1:])):
        raise ValueError("Private OCR cannot load all language/orientation packs")
    print(f"Bundled {version}: {len(libraries)} libraries, eng/fra/ara/osd", flush=True)
    return manifest
