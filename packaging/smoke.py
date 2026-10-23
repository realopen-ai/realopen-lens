"""Exercise the frozen executable, including subprocess indexing and HTTP gallery.

Run with an existing verified model directory; all generated data stays temporary.
"""

import argparse
import hashlib
from http.cookiejar import CookieJar
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from PIL import Image, ImageDraw


def verify_search(results, source):
    if len(results) != 1 or source.resolve() not in {
        Path(path).resolve() for path in results[0]["paths"]
    }:
        raise ValueError("Search did not return exactly the fixture image")
    result = results[0]
    if result["status"] != "ready" or result["issues"]:
        raise ValueError("Search returned degraded data")
    if not all(word in result["text"].lower() for word in ("invoice", "abc-123", "pizza")):
        raise ValueError("OCR did not recognize the fixture reference text")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--siglip-precision", choices=("int8", "fp32"), default="int8")
    args = parser.parse_args()
    executable, models = args.executable.resolve(), args.models.resolve()
    environment = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    # Only the app's private engine/data may satisfy frozen inference acceptance.
    environment["PATH"] = ""
    environment["TESSDATA_PREFIX"] = "/nonexistent-host-tessdata"
    with tempfile.TemporaryDirectory(prefix="lens-frozen-") as directory:
        base = Path(directory)
        images, index = base / "images", base / "index"
        images.mkdir()
        image = Image.new("RGB", (640, 320), "white")
        ImageDraw.Draw(image).text((30, 60), "Invoice ABC-123 pizza", fill="black", font_size=32)
        image.save(images / "invoice.png")
        original = hashlib.sha256((images / "invoice.png").read_bytes()).hexdigest()

        def cli(*arguments):
            result = subprocess.run(
                [str(executable), "--cli", *map(str, arguments)],
                cwd=base,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=180,
            )
            if result.returncode:
                raise RuntimeError(f"{arguments}: {result.stdout}\n{result.stderr}")
            return json.loads(result.stdout)

        indexed = cli(
            "index",
            images,
            "--index-dir",
            index,
            "--models",
            models,
            "--languages",
            "eng+fra+ara",
            "--siglip-precision",
            args.siglip_precision,
        )
        assert indexed["status"] == "completed" and indexed["images"] == 1, indexed
        assert indexed["failed_files"] == indexed["degraded_files"] == 0, indexed
        searches = {}
        for mode in ("lexical", "hybrid"):
            searched = cli(
                "search", "invoice", "--mode", mode, "--index-dir", index, "--models", models
            )
            verify_search(searched, images / "invoice.png")
            searches[mode] = searched
        refreshed = cli(
            "index",
            images,
            "--index-dir",
            index,
            "--models",
            models,
            "--languages",
            "eng+fra+ara",
            "--siglip-precision",
            args.siglip_precision,
        )
        assert refreshed["status"] == "completed" and refreshed["reused_images"] == 1, refreshed
        resident = subprocess.run(
            [str(executable), "-m", "realopen_lens.runtime.query_server"],
            cwd=base,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            input="".join(
                json.dumps(
                    {
                        "models": str(models),
                        "kind": kind,
                        "threads": 2,
                        "siglip_precision": args.siglip_precision,
                        "query": "invoice",
                    }
                )
                + "\n"
                for kind in ("siglip2", "e5")
            ),
        )
        assert resident.returncode == 0, resident.stderr
        vectors = [json.loads(line)["vector"] for line in resident.stdout.splitlines()]
        assert [len(vector) for vector in vectors] == [768, 384]
        assert all(any(v) and all(math.isfinite(x) for x in v) for v in vectors)
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        with (base / "server.log").open("w") as log:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
            process = subprocess.Popen(
                [
                    str(executable),
                    "--cli",
                    "serve",
                    "--index-dir",
                    str(index),
                    "--models",
                    str(models),
                    "--port",
                    str(port),
                ],
                cwd=base,
                env=environment,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError((base / "server.log").read_text())
                    try:
                        with opener.open(f"http://127.0.0.1:{port}", timeout=1) as r:
                            assert b"<html" in r.read()
                        with opener.open(f"http://127.0.0.1:{port}/api/session", timeout=1) as r:
                            assert json.load(r)["allowed_roots"] == [str(Path.home().resolve())]
                        with opener.open(f"http://127.0.0.1:{port}/api/folders", timeout=5) as r:
                            folders = json.load(r)
                            assert folders["path"] == str(Path.home().resolve())
                            assert isinstance(folders["folders"], list)
                        other = images / "another-folder"
                        other.mkdir()
                        request = urllib.request.Request(
                            f"http://127.0.0.1:{port}/api/tasks/index",
                            data=json.dumps(
                                {"root": str(other), "collection": "personal"}
                            ).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        try:
                            opener.open(request, timeout=5)
                        except urllib.error.HTTPError as error:
                            assert error.code == 409
                            assert "Use another collection name" in json.load(error)["detail"]
                        else:
                            raise AssertionError("Collection-name conflict was queued")
                        break
                    except OSError:
                        time.sleep(0.2)
                else:
                    raise TimeoutError("Frozen server did not become ready")
                print(
                    json.dumps(
                        {
                            "index": indexed,
                            "search": searches,
                            "refresh": refreshed,
                            "siglip_precision": args.siglip_precision,
                            "source_unchanged": hashlib.sha256(
                                (images / "invoice.png").read_bytes()
                            ).hexdigest()
                            == original,
                            "gallery": "passed",
                            "resident_dimensions": [len(v) for v in vectors],
                        }
                    )
                )
            finally:
                if sys.platform == "win32":
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).with_name("windows_signal.py")),
                                str(process.pid),
                            ],
                            check=True,
                            timeout=10,
                            capture_output=True,
                        )
                    except Exception:
                        process.kill()
                        process.wait(timeout=5)
                        raise
                else:
                    process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise TimeoutError("Frozen gallery required forced shutdown")
                expected = (0, 3221225786, -1073741510) if sys.platform == "win32" else (0, -15)
                assert process.returncode in expected, "Frozen gallery did not shut down cleanly"
                assert "Application shutdown complete" in (base / "server.log").read_text(
                    encoding="utf-8"
                ), "Frozen gallery shutdown did not complete"
        assert hashlib.sha256((images / "invoice.png").read_bytes()).hexdigest() == original


if __name__ == "__main__":
    main()
