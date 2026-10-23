"""Build a native Linux x64 portable tarball with private OCR."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tesseract", type=Path, help="Build-time Tesseract 5 executable")
    parser.add_argument("--tessdata-dir", type=Path, default=Path("build/tessdata"))
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("Build Linux x64 artifacts on Linux x64")
    root = Path(__file__).resolve().parents[1]
    source = args.tesseract or shutil.which("tesseract")
    if not source or not shutil.which("patchelf"):
        raise SystemExit("Install Tesseract 5 and patchelf on the Linux build machine")
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}

    def run(*command):
        subprocess.run(command, cwd=root, env=environment, check=True, timeout=1800)

    run("npm", "--prefix", "frontend", "run", "build")
    run(sys.executable, "packaging/icons.py")
    with tempfile.TemporaryDirectory(prefix="lens-linux-build-") as directory:
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
        from bundle_linux_ocr import bundle_ocr

        bundle_ocr(bundle, Path(source).resolve(), args.tessdata_dir.resolve())
        shutil.copyfile(root / "packaging/linux-README.txt", bundle / "README.txt")
        destination = root / "dist/RealOpenLens-linux-x64.tar.gz"
        destination.parent.mkdir(exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        # tar preserves executable bits and the relative links PyInstaller uses.
        with tarfile.open(partial, "w:gz", dereference=False) as archive:
            archive.add(bundle, arcname=bundle.name)
        partial.replace(destination)
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        destination.with_name(destination.name + ".sha256").write_text(
            f"{digest}  {destination.name}\n", encoding="utf-8", newline="\n"
        )
        print(f"Linux portable archive: {destination}", flush=True)


if __name__ == "__main__":
    main()
