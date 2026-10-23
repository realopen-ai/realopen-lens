"""Build a native Windows x64 portable ZIP; never cross-compile."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcpkg-root", type=Path, required=True, help="Pinned bootstrapped vcpkg")
    parser.add_argument("--tessdata-dir", type=Path, help="Reuse matching pinned language files")
    return parser.parse_args(argv)


def main():
    args = options()
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit("Build Windows x64 artifacts on Windows x64")
    root = Path(__file__).resolve().parents[1]
    manifest = root / "packaging/windows-ocr"
    expected = json.loads((manifest / "vcpkg.json").read_text("utf-8"))["builtin-baseline"]
    vcpkg = args.vcpkg_root.resolve()
    actual = subprocess.run(
        ["git", "-C", str(vcpkg), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    if actual != expected:
        raise SystemExit(f"Use vcpkg revision {expected}; found {actual}")
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "VCPKG_DISABLE_METRICS": "1"}
    installed = root / "build/windows-ocr-installed"

    def run(*command, timeout=1800):
        subprocess.run(command, cwd=root, env=environment, check=True, timeout=timeout)

    run(
        str(vcpkg / "vcpkg.exe"),
        "install",
        f"--x-manifest-root={manifest}",
        f"--x-install-root={installed}",
        f"--overlay-triplets={manifest / 'triplets'}",
        "--triplet=lens-x64-windows-static",
        timeout=3600,
    )
    npm = shutil.which("npm.cmd")
    if not npm:
        raise SystemExit("Install Node.js/npm on the Windows build machine")
    run(npm, "--prefix", "frontend", "run", "build")
    run(sys.executable, "packaging/icons.py")
    with tempfile.TemporaryDirectory(prefix="lens-windows-build-") as directory:
        run(
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            directory,
            "packaging/desktop.spec",
        )
        bundle = Path(directory) / "RealOpenLens"
        from bundle_windows_ocr import bundle_ocr

        bundle_ocr(bundle, installed / "lens-x64-windows-static", args.tessdata_dir)
        shutil.copyfile(root / "packaging/windows-README.txt", bundle / "README.txt")
        destination = root / "dist/RealOpenLens-windows-x64.zip"
        destination.parent.mkdir(exist_ok=True)
        partial = destination.with_suffix(".zip.partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle.parent).as_posix())
        partial.replace(destination)
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        destination.with_suffix(".zip.sha256").write_text(
            f"{digest}  {destination.name}\n",
            encoding="utf-8",
        )
        print(f"Windows portable archive: {destination}")


if __name__ == "__main__":
    main()
