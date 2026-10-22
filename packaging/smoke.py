"""Exercise the frozen executable, including subprocess indexing and HTTP gallery.

Run with an existing verified model directory; all generated data stays temporary.
"""

import argparse
from http.cookiejar import CookieJar
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    executable, models = args.executable.resolve(), args.models.resolve()
    with tempfile.TemporaryDirectory(prefix="lens-frozen-") as directory:
        base = Path(directory)
        images, index = base / "images", base / "index"
        images.mkdir()
        image = Image.new("RGB", (640, 320), "white")
        ImageDraw.Draw(image).text((30, 60), "Invoice ABC-123 pizza", fill="black", font_size=32)
        image.save(images / "invoice.png")

        def cli(*arguments):
            result = subprocess.run(
                [str(executable), "--cli", *map(str, arguments)],
                cwd=base,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode:
                raise RuntimeError(f"{arguments}: {result.stdout}\n{result.stderr}")
            return json.loads(result.stdout)

        indexed = cli(
            "index", images, "--index-dir", index, "--models", models, "--languages", "eng"
        )
        assert indexed["images"] == 1 and indexed["failed_files"] == 0, indexed
        searched = cli("search", "invoice", "--index-dir", index, "--models", models)
        assert searched, searched
        resident = subprocess.run(
            [str(executable), "-m", "realopen_lens.runtime.query_server"],
            cwd=base,
            capture_output=True,
            text=True,
            timeout=120,
            input="".join(
                json.dumps(
                    {
                        "models": str(models),
                        "kind": kind,
                        "threads": 2,
                        "siglip_precision": "int8",
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
                stdout=log,
                stderr=log,
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
                            "search": searched,
                            "gallery": "passed",
                            "resident_dimensions": [len(v) for v in vectors],
                        }
                    )
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
