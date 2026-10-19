"""Small adversarial input probes, separate from the 500 valid-image collection."""

import argparse
import hashlib
import io
import json
import struct
import zlib
from pathlib import Path
from typing import Any
from time import perf_counter
import psutil
from PIL import Image
from realopen_lens.encoders.onnx import image_rgb


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def run(data: Path, output: Path) -> list[dict]:
    image = Image.new("RGB", (80, 40), "#3276a9")
    png = io.BytesIO()
    image.save(png, format="PNG")
    gif = io.BytesIO()
    image.save(gif, format="GIF")
    jpeg = io.BytesIO()
    image.save(jpeg, format="JPEG")
    tiff = io.BytesIO()
    image.save(
        tiff, format="TIFF", save_all=True, append_images=[Image.new("RGB", (80, 40), "red")]
    )
    bomb = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 100000, 100000, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(b"\0"))
        + png_chunk(b"IEND", b"")
    )
    cases = [
        ("valid.png", png.getvalue(), "success"),
        ("wrong-extension.jpg", png.getvalue(), "success"),
        ("truncated.png", png.getvalue()[:20], "reject"),
        ("truncated.jpg", jpeg.getvalue()[:100], "reject"),
        ("html.png", b"<html>This is not an image.</html>", "reject"),
        ("pixel-bomb.png", bomb, "reject"),
        ("multipage.tif", tiff.getvalue(), "first-frame"),
        ("empty.webp", b"", "reject"),
        ("unsupported-gif.png", gif.getvalue(), "reject"),
    ]
    data.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, payload, expected in cases:
        path = data / name
        sha = hashlib.sha256(payload).hexdigest()
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"Refusing to replace modified probe: {path}")
        path.write_bytes(payload)
        start = perf_counter()
        rss = psutil.Process().memory_info().rss
        record: dict[str, Any] = {
            "id": name,
            "sha256": sha,
            "expected": expected,
            "stage": "decode",
            "error": None,
            "source": "Authored deterministic safety probe; CC0-1.0",
        }
        try:
            with image_rgb(path) as decoded:
                pixel = decoded.getpixel((0, 0))
                assert isinstance(pixel, tuple)
                record["actual"] = {
                    "size": list(decoded.size),
                    "mode": decoded.mode,
                    "first_pixel": list(pixel),
                }
            record["passed"] = expected in ("success", "first-frame") and record["actual"][
                "size"
            ] == [80, 40]
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            record.update(
                actual="rejected",
                error={"type": type(error).__name__, "message": str(error)},
                passed=expected == "reject",
            )
        record.update(
            seconds=perf_counter() - start,
            observed_rss_mib=max(rss, psutil.Process().memory_info().rss) / 2**20,
            memory_method="Two RSS snapshots, not a measured peak",
        )
        records.append(record)
    destination = output / "individual.jsonl"
    if destination.exists():
        raise ValueError("Probe output already exists; choose a new output directory")
    destination.write_text("".join(json.dumps(r) + "\n" for r in records))
    (output / "summary.json").write_text(
        json.dumps(
            {
                "cases": len(records),
                "passed": sum(r["passed"] for r in records),
                "records": str(destination),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Safety probes: {sum(r['passed'] for r in records)}/{len(records)} passed")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/benchmark-probes"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
