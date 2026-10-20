"""Render fixed identifier diagnostics without downloads or platform-specific fonts."""

import argparse
from io import BytesIO
from pathlib import Path
import json

from PIL import Image, ImageDraw, ImageFont, __version__, features

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json


def build(output: Path) -> dict:
    source = Path(__file__).resolve().parents[1] / "sources/numeric-identifiers-v1/documents.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=28)
    documents = []
    for row in manifest["documents"]:
        with Image.new("RGB", (1100, 700), "white") as image:
            ImageDraw.Draw(image).multiline_text(
                (60, 70), row["reference_text"], font=font, fill="black", spacing=28
            )
            buffer = BytesIO()
            image.save(buffer, format="PNG")
        path = output / row["path"]
        payload = buffer.getvalue()
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Refusing to replace a different fixture; use a new output directory")
        if not path.exists():
            path.write_bytes(payload)
        actual = digest(path)
        documents.append(
            dict(
                row,
                sha256=actual,
                reference_sha256=row["sha256"],
                matches_reference_bytes=actual == row["sha256"],
            )
        )
    result = dict(
        manifest,
        documents=documents,
        pillow_version=__version__,
        freetype_version=features.version("freetype2"),
        font="Pillow embedded default, 28px",
    )
    save_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
