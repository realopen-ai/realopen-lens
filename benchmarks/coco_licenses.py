"""Read COCO's official per-image licenses without fetching the 253 MB archive."""

import io
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip"
SIZE = 252872794
ETAG = '"0a379cfc70b0e71301e0f377548639bd"'


class RangeReader(io.RawIOBase):
    def __init__(self) -> None:
        self.position = 0

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (
            offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        )
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        stop = SIZE if size < 0 else min(SIZE, self.position + size)
        if stop <= self.position:
            return b""
        result = subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "2",
                "--max-time",
                "90",
                "--range",
                f"{self.position}-{stop-1}",
                "-H",
                f"If-Match: {ETAG}",
                URL,
            ],
            check=True,
            capture_output=True,
        ).stdout
        if len(result) != stop - self.position:
            raise ValueError("Server did not honor pinned COCO archive byte range")
        self.position = stop
        return result


def load(cache: Path) -> dict:
    path = cache / "coco-license-info.json"
    if not path.exists():
        with zipfile.ZipFile(RangeReader()) as archive:
            content = json.loads(archive.read("annotations/captions_val2014.json"))
        path.write_text(
            json.dumps(
                {
                    "licenses": content["licenses"],
                    "images": content["images"],
                    "source": URL,
                    "archive_etag": ETAG,
                }
            ),
            encoding="utf-8",
        )
    lock = json.loads((Path(__file__).parent / "sources/coco-license-info.lock.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != lock["sha256"]:
        raise ValueError("COCO license metadata checksum mismatch")
    return json.loads(path.read_text())
