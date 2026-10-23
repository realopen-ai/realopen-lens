"""Validate the distributed archive and retain checksum-bound per-stage evidence."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import zipfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(archive, output, models=None, precisions=("int8", "fp32"), require_notarized=False):
    if sys.platform not in {"darwin", "win32"}:
        raise ValueError("Validate desktop archives on macOS or Windows")
    windows = sys.platform == "win32"
    if windows and require_notarized:
        raise ValueError("Notarization checks apply only to macOS archives")
    archive, output = archive.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    report = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "archive": archive.name,
        "archive_sha256": None,
        "host": platform.platform(),
        "checks": [],
        "scope": "existing host with temporary workspaces; not a clean OS installation",
        "inference": "requested" if models else "not_run",
        "notarization": (
            "not_applicable" if windows else ("required" if require_notarized else "not_checked")
        ),
        "status": "failed",
        "results": {},
    }
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def run(name, command, timeout):
        record = {"name": name, "status": "failed"}
        report["checks"].append(record)
        started = time.monotonic()
        print(f"[{len(report['checks'])}] {name}", flush=True)
        try:
            # Stream output to bounded-per-command logs; do not buffer inference output in RAM.
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
                )
            if completed.returncode:
                raise RuntimeError(f"{name} exited {completed.returncode}; see {name}.log")
            record["status"] = "passed"
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["seconds"] = time.monotonic() - started

    try:
        report["archive_sha256"] = digest(archive)
        run(
            "package",
            [
                sys.executable,
                str(root / "ci_smoke.py"),
                str(archive),
                "--report",
                str(output / "package.json"),
            ],
            300,
        )
        if models or require_notarized:
            with tempfile.TemporaryDirectory(prefix="lens-acceptance-") as directory:
                if windows:
                    with zipfile.ZipFile(archive) as zipped:
                        destination = Path(directory).resolve()
                        for item in zipped.infolist():
                            if (
                                not (destination / item.filename)
                                .resolve()
                                .is_relative_to(destination)
                            ):
                                raise ValueError("Archive member escapes extraction directory")
                        zipped.extractall(destination)
                    executable = Path(directory) / "RealOpenLens/RealOpenLens.exe"
                else:
                    run("extract", ["/usr/bin/ditto", "-x", "-k", str(archive), directory], 60)
                    executable = Path(directory) / "RealOpen Lens.app/Contents/MacOS/RealOpenLens"
                if require_notarized:
                    bundle = executable.parents[2]
                    run(
                        "stapled-ticket", ["/usr/bin/xcrun", "stapler", "validate", str(bundle)], 60
                    )
                    run(
                        "gatekeeper",
                        [
                            "/usr/sbin/spctl",
                            "--assess",
                            "--type",
                            "execute",
                            "--verbose=2",
                            str(bundle),
                        ],
                        60,
                    )
                    report["notarization"] = "passed"
                for precision in precisions if models else ():
                    assert models is not None
                    run(
                        f"inference-{precision}",
                        [
                            sys.executable,
                            str(root / "smoke.py"),
                            str(executable),
                            "--models",
                            str(models.resolve()),
                            "--siglip-precision",
                            precision,
                        ],
                        1200,
                    )
                    result = json.loads(
                        (output / f"inference-{precision}.log")
                        .read_text(encoding="utf-8")
                        .splitlines()[-1]
                    )
                    if result.get("siglip_precision") != precision:
                        raise ValueError(
                            "Inference output does not identify the requested precision"
                        )
                    report["results"][precision] = result
            if models:
                report["inference"] = "passed"
        if digest(archive) != report["archive_sha256"]:
            raise ValueError("Archive changed during acceptance")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "acceptance.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New evidence directory for this run"
    )
    parser.add_argument(
        "--models", type=Path, help="Existing models; enables actual inference checks"
    )
    parser.add_argument("--precision", choices=("int8", "fp32", "both"), default="both")
    parser.add_argument(
        "--require-notarized",
        action="store_true",
        help="Require a valid stapled ticket and Gatekeeper acceptance",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to preserve historical evidence")
    precisions = ("int8", "fp32") if args.precision == "both" else (args.precision,)
    validate(args.archive, args.output, args.models, precisions, args.require_notarized)


if __name__ == "__main__":
    main()
