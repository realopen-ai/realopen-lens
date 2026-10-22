"""Build a development archive or explicitly signed, notarized macOS release."""

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
import tomllib


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="Require signing and notarization")
    parser.add_argument("--identity", help="Installed Developer ID Application identity")
    parser.add_argument("--notary-profile", help="Existing notarytool Keychain credential profile")
    parser.add_argument("--tesseract", type=Path, help="Build-time Tesseract 5 executable")
    parser.add_argument(
        "--tessdata-dir", type=Path, help="Existing language data to reuse if checksums match"
    )
    args = parser.parse_args(argv)
    if args.release and (
        not args.identity
        or not args.identity.startswith("Developer ID Application:")
        or not args.notary_profile
    ):
        parser.error(
            "--release requires --identity 'Developer ID Application: …' and --notary-profile"
        )
    if not args.release and (args.identity or args.notary_profile):
        parser.error("Signing options require --release; partial release signing is not supported")
    return args


def main():
    args = options()
    if sys.platform != "darwin":
        raise SystemExit("Build macOS artifacts on macOS")
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    # Do not inherit a signing identity from an unrelated shell/build.
    environment.pop("LENS_CODESIGN_IDENTITY", None)
    if args.release:
        environment["LENS_CODESIGN_IDENTITY"] = args.identity
        environment["PYINSTALLER_STRICT_BUNDLE_CODESIGN_ERROR"] = "1"

    def run(*command):
        subprocess.run(command, cwd=root, env=environment, check=True, timeout=1800)

    if args.release:
        identities = subprocess.run(
            ["/usr/bin/security", "find-identity", "-v", "-p", "codesigning"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        if args.identity not in identities:
            raise SystemExit("Requested Developer ID Application identity is not installed")

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
        from bundle_ocr import bundle_ocr

        source_ocr = args.tesseract or shutil.which("tesseract")
        if not source_ocr:
            raise SystemExit("Install Tesseract 5 on the build machine or provide --tesseract")
        bundle_ocr(bundle, Path(source_ocr), args.tessdata_dir, args.identity)
        run("/usr/bin/codesign", "--verify", "--deep", "--strict", str(bundle))
        version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "version"
        ]
        suffix = f"-{version}" if args.release else ""
        archive = root / "dist" / f"RealOpenLens{suffix}-macos-{platform.machine()}.zip"
        archive.parent.mkdir(exist_ok=True)

        def compress(destination):
            run(
                "/usr/bin/ditto",
                "-c",
                "-k",
                "--norsrc",
                "--noextattr",
                "--keepParent",
                str(bundle),
                str(destination),
            )

        if args.release:
            upload = Path(output) / "notarization.zip"
            compress(upload)
            result = subprocess.run(
                [
                    "/usr/bin/xcrun",
                    "notarytool",
                    "submit",
                    str(upload),
                    "--keychain-profile",
                    args.notary_profile,
                    "--wait",
                    "--timeout",
                    "20m",
                    "--output-format",
                    "json",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=1260,
            )
            try:
                notarization = json.loads(result.stdout)
            except json.JSONDecodeError:
                notarization = {"status": "Error", "output": result.stdout, "error": result.stderr}
            (root / "build/notarization.json").write_text(
                json.dumps(notarization, indent=2) + "\n", encoding="utf-8"
            )
            if result.returncode or notarization.get("status") != "Accepted":
                raise SystemExit("Notarization was not accepted; see build/notarization.json")
            run("/usr/bin/xcrun", "stapler", "staple", str(bundle))
            run("/usr/bin/xcrun", "stapler", "validate", str(bundle))
            run("/usr/sbin/spctl", "--assess", "--type", "execute", "--verbose=2", str(bundle))
        # Replace the local deliverable only after every requested gate succeeds.
        staged = Path(output) / "deliverable.zip"
        compress(staged)
        publishing = archive.with_suffix(".zip.partial")
        shutil.copyfile(staged, publishing)
        publishing.replace(archive)
        with archive.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        archive.with_suffix(".zip.sha256").write_text(
            f"{sha256}  {archive.name}\n", encoding="utf-8"
        )
        print(f"{'Notarized release' if args.release else 'Development archive'}: {archive}")


if __name__ == "__main__":
    main()
