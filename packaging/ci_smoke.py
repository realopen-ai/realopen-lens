"""Model-free checks against an extracted macOS development archive."""

import argparse
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from PIL import Image


def validate(archive: Path, report_path: Path):
    report = {"archive": archive.name, "host": platform.platform(), "checks": []}
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def check(name, action, verify=None):
        started = time.monotonic()
        record = {"name": name, "status": "failed"}
        report["checks"].append(record)
        try:
            result = action()
            if verify:
                verify(result)
            record["status"] = "passed"
            return result
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["seconds"] = time.monotonic() - started

    def run(command, *, cwd, data=None, expected=0):
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            input=data,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != expected:
            raise RuntimeError(
                f"{command}: exit {result.returncode}\n"
                f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
            )
        return result

    try:
        with tempfile.TemporaryDirectory(prefix="lens-ci-smoke-") as directory:
            base = Path(directory)
            check(
                "extract archive",
                lambda: run(["/usr/bin/ditto", "-x", "-k", str(archive), str(base)], cwd=base),
            )
            bundle = base / "RealOpen Lens.app"
            executable = bundle / "Contents/MacOS/RealOpenLens"
            check(
                "ad-hoc signature",
                lambda: run(
                    ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(bundle)], cwd=base
                ),
            )

            def icon():
                contents = bundle / "Contents"
                info = plistlib.loads((contents / "Info.plist").read_bytes())
                assert info["CFBundleIdentifier"] == "org.realopen.lens"
                license_text = (contents / "Resources/licenses/realopen-lens/LICENSE").read_text(
                    encoding="utf-8"
                )
                assert "Apache License" in license_text and "Version 2.0" in license_text
                notices = (
                    contents / "Resources/licenses/realopen-lens/THIRD_PARTY_NOTICES.md"
                ).read_text(encoding="utf-8")
                assert notices.startswith("# Third-party notices")
                assert all(
                    name in notices for name in ("CPython", "onnxruntime", "Tcl", "Tk", "react")
                )
                with Image.open(contents / "Resources" / info["CFBundleIconFile"]) as image:
                    image = image.convert("RGBA")
                    assert image.size == (1024, 1024)
                    bounds = image.getbbox()
                    assert bounds and 90 <= bounds[0] <= 110 and bounds[2] <= 934
                    assert image.getpixel((0, 0))[3] == 0

            check("icon, license and bundle metadata", icon)
            diagnostics = check(
                "frozen dependencies and Tcl/Tk resources",
                lambda: json.loads(run([str(executable), "--check-bundle"], cwd=base).stdout),
                verify=lambda value: require(
                    value["frozen"] and value["tk_resources"],
                    "Bundle diagnostics did not confirm frozen Tk resources",
                ),
            )
            report["bundle"] = diagnostics
            check(
                "stage worker dispatch",
                lambda: run(
                    [str(executable), "-m", "realopen_lens.runtime.worker", "--help"], cwd=base
                ).stdout,
                verify=lambda value: require(
                    "--image-query" in value and "--request" in value,
                    "Missing stage worker CLI flags",
                ),
            )
            check(
                "resident worker clean EOF",
                lambda: run(
                    [str(executable), "-m", "realopen_lens.runtime.query_server"], cwd=base, data=""
                ),
            )
            check(
                "resident worker structured error",
                lambda: json.loads(
                    run(
                        [str(executable), "-m", "realopen_lens.runtime.query_server"],
                        cwd=base,
                        data="{}\n",
                        expected=1,
                    ).stdout
                ),
                verify=lambda value: require(
                    value["error"].startswith("KeyError:") and "models" in value["error"],
                    "Resident worker did not return the expected protocol error",
                ),
            )
            check(
                "unknown worker refused",
                lambda: run([str(executable), "-m", "http.server"], cwd=base, expected=1),
                verify=lambda value: require(
                    "Unsupported Lens worker" in value.stderr,
                    "Unrecognized worker was not explicitly refused",
                ),
            )
            index = base / "index"
            check(
                "frozen CLI workspace info",
                lambda: json.loads(
                    run(
                        [str(executable), "--cli", "workspace", "info", "--index-dir", str(index)],
                        cwd=base,
                    ).stdout
                ),
                verify=lambda value: require(
                    not value["initialized"] and value["build"]["gallery_available"],
                    "Frozen workspace CLI has missing gallery or unexpected data",
                ),
            )

            def server():
                images = base / "images"
                images.mkdir()
                with socket.socket() as reserved:
                    reserved.bind(("127.0.0.1", 0))
                    port = reserved.getsockname()[1]
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({}),
                    urllib.request.HTTPCookieProcessor(CookieJar()),
                )
                log_path = report_path.with_name("desktop-ci-server.log")
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        [
                            str(executable),
                            "--cli",
                            "serve",
                            "--index-dir",
                            str(index),
                            "--models",
                            str(base / "no-models"),
                            "--allow-root",
                            str(images),
                            "--port",
                            str(port),
                        ],
                        cwd=base,
                        env={**environment, "PATH": str(base / "no-external-tools")},
                        stdout=log,
                        stderr=log,
                    )
                    try:
                        url = f"http://127.0.0.1:{port}"
                        deadline = time.monotonic() + 30
                        while True:
                            if process.poll() is not None:
                                raise RuntimeError(log_path.read_text(encoding="utf-8"))
                            try:
                                with opener.open(url, timeout=1) as response:
                                    html = response.read().decode("utf-8")
                                break
                            except urllib.error.HTTPError:
                                raise
                            except OSError:
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("Frozen server startup timed out")
                                time.sleep(0.2)
                        assert "<html" in html
                        assets = set(re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html))
                        assert "/assets/lens.svg" in assets
                        for asset in assets:
                            with opener.open(url + asset, timeout=5) as response:
                                assert response.read(), asset
                        with opener.open(url + "/api/session", timeout=5) as response:
                            assert json.load(response)["allowed_roots"] == [str(images.resolve())]
                        with opener.open(url + "/api/workspace", timeout=5) as response:
                            workspace = json.load(response)
                            assert workspace["directory"] == str(index.resolve())
                            assert (
                                workspace["build"]["code_sha256"]
                                == diagnostics["build"]["code_sha256"]
                            )
                        with opener.open(url + "/api/collections", timeout=5) as response:
                            assert json.load(response) == []
                        with opener.open(url + "/api/folders", timeout=5) as response:
                            assert json.load(response)["path"] == str(images.resolve())
                        request = urllib.request.Request(
                            url + "/api/tasks/setup",
                            data=b"{}",
                            headers={"Content-Type": "application/json"},
                        )
                        with opener.open(request, timeout=5) as response:
                            task = json.load(response)
                        deadline = time.monotonic() + 30
                        while task["status"] in {"queued", "running"}:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Frozen setup check timed out")
                            with opener.open(
                                url + "/api/tasks/" + task["id"], timeout=5
                            ) as response:
                                task = json.load(response)
                            time.sleep(0.1)
                        assert task["status"] == "completed", task
                        assert task["result"]["ready"] is False
                        assert not next(c for c in task["result"]["checks"] if c["id"] == "models")[
                            "ok"
                        ]
                        assert not next(c for c in task["result"]["checks"] if c["id"] == "ocr")[
                            "ok"
                        ]
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                # Uvicorn restores and re-raises SIGTERM after successful cleanup.
                assert process.returncode in (0, -signal.SIGTERM), process.returncode
                assert "Application shutdown complete." in log_path.read_text(encoding="utf-8")

            check("loopback gallery, authenticated API, assets and clean shutdown", server)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--report", type=Path, default=Path("build/desktop-ci-report.json"))
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = validate(args.archive.resolve(), args.report.resolve())
    print(f"Passed {len(result['checks'])} frozen-package checks without model downloads")


if __name__ == "__main__":
    main()
