"""Deterministic text-presence controls; benchmark infrastructure, never production."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from realopen_lens.models import digest


def build(output: Path, dataset: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    font_spec = json.loads(Path("benchmarks/sources/font.json").read_text())
    from benchmarks.dataset import download

    font = download(
        font_spec["url"], Path(".cache/hardening/NotoSans.ttf"), font_spec["git_blob_sha1"]
    )
    shutil.copyfile(font, output / "font.ttf")
    rows: list[dict] = []

    def save(image, name, category, truth, difficulty, source):
        path = output / (name + ".png")
        image.save(path)
        rows.append(
            {
                "id": name,
                "path": path.name,
                "category": category,
                "subcategory": "presence-control",
                "width": image.width,
                "height": image.height,
                "format": "PNG",
                "sha256": digest(path),
                "source": source,
                "split": "dev" if len(rows) % 3 else "holdout",
                "difficulty": difficulty,
                "expected": {"decode": "success"},
                "truth": truth,
                "language": "en",
                "negative_text": not bool(truth),
                "family": name,
            }
        )

    for n in range(18):
        small = n % 3 == 0
        faint = n % 3 == 1
        dark = n % 3 == 2
        size = (1600, 900) if small else (900, 500)
        text = f"REGION-{n:02d} connexion PostgreSQL 5432"
        with Image.new("RGB", size, "#202020" if dark else "white") as im:
            d = ImageDraw.Draw(im)
            f = ImageFont.truetype(str(font), 12 if small else 22)
            xy = (12, 12) if n < 9 else (size[0] // 3, size[1] // 2)
            d.text(xy, text, font=f, fill="#cccccc" if faint else ("white" if dark else "black"))
            if n >= 12:
                d.rectangle((xy[0] - 5, xy[1] - 5, xy[0] + 500, xy[1] + 40), outline="#888888")
            save(
                im,
                f"positive-{n:02d}",
                "screenshots",
                text,
                ["small-text" if small else "faint-text" if faint else "dark-background"],
                {"kind": "synthetic", "license": "CC0-1.0", "seed": 20261019},
            )
    rng = np.random.default_rng(20261019)
    for n in range(12):
        with Image.new("RGB", (600, 600), ["white", "#808080", "black"][n % 3]) as im:
            d = ImageDraw.Draw(im)
            if n >= 3:
                for _ in range(12):
                    x, y = rng.integers(10, 500, size=2).tolist()
                    d.ellipse((x, y, x + 50, y + 40), outline="#555555", width=2)
            save(
                im,
                f"negative-{n:02d}",
                "challenging",
                "",
                ["known-no-text"],
                {"kind": "synthetic", "license": "CC0-1.0", "seed": 20261019},
            )
    source_rows = {
        r["id"]: r for r in json.loads((dataset / "manifest.json").read_text())["images"]
    }
    for n, number in enumerate(["100661", "107814", "229529", "266977", "434463", "451683"]):
        name = f"photos/COCO_val2014_000000{number}.jpg"
        r = source_rows[name]
        assert digest(dataset / r["path"]) == r["sha256"]
        with Image.open(dataset / r["path"]) as im:
            save(
                im.convert("RGB"),
                f"real-negative-{n:02d}",
                "photos",
                "",
                ["manually-reviewed-no-text"],
                {
                    **r["source"],
                    "kind": "real",
                    "parent": name,
                    "annotation": (
                        "Reviewed full image: no visible readable text. " "Scene words are not OCR."
                    ),
                },
            )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 20261019,
                "font_sha256": digest(output / "font.ttf"),
                "font_source": font_spec,
                "images": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Created {len(rows)} controls in {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/text-region-controls-v1"))
    p.add_argument("--dataset", type=Path, default=Path("data/benchmark-500-v2"))
    a = p.parse_args()
    build(a.output, a.dataset)


if __name__ == "__main__":
    main()
