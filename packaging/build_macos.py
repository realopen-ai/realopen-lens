"""Freeze in clean temporary storage, verify ad-hoc signing and archive the app."""

import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def main():
    if sys.platform != "darwin":
        raise SystemExit("Build macOS artifacts on macOS")
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}

    def run(*command):
        subprocess.run(command, cwd=root, env=environment, check=True)

    run("npm", "--prefix", "frontend", "run", "build")
    run(sys.executable, "packaging/icons.py")
    # File-provider metadata in a checkout can invalidate an otherwise valid signature.
    with tempfile.TemporaryDirectory(prefix="lens-desktop-build-") as output:
        run(
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            output,
            "packaging/desktop.spec",
        )
        bundle = Path(output) / "RealOpen Lens.app"
        run("/usr/bin/codesign", "--verify", "--deep", "--strict", str(bundle))
        archive = root / "dist" / f"RealOpenLens-macos-{platform.machine()}.zip"
        archive.parent.mkdir(exist_ok=True)
        run(
            "/usr/bin/ditto",
            "-c",
            "-k",
            "--norsrc",
            "--noextattr",
            "--keepParent",
            str(bundle),
            str(archive),
        )
        print(f"Development archive: {archive}")


if __name__ == "__main__":
    main()
