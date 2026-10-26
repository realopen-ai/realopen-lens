"""Build deterministic, explicitly synthetic multilingual OCR diagnostic images."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import urllib.request

from PIL import Image, ImageFilter

SOURCE = Path(__file__).resolve().parents[1] / "sources/multilingual-ocr-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_ocr_assets(assets: Path):
    registry = json.loads(
        (SOURCE.parents[2] / "packaging/tessdata.json").read_text(encoding="utf-8")
    )
    for code, spec in registry.items():
        path = assets / "tessdata" / f"{code}.traineddata"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or digest(path) != spec["sha256"]:
            with urllib.request.urlopen(spec["url"], timeout=90) as response:
                data = response.read(spec["size"] + 1)
            if len(data) != spec["size"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError(f"Corrupt language pack: {code}")
            path.write_bytes(data)
    config = assets / "tessdata/configs/tsv"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("tessedit_create_tsv 1\n", encoding="utf-8")


def build(output: Path, assets: Path, *, fetch: bool = False) -> list[dict]:
    if fetch:
        prepare_ocr_assets(assets)
    fonts = json.loads((SOURCE / "fonts.json").read_text(encoding="utf-8"))
    assets.mkdir(parents=True, exist_ok=True)
    for spec in fonts.values():
        path = assets / spec["file"]
        if fetch and not path.is_file():
            with urllib.request.urlopen(spec["url"], timeout=90) as response:
                data = response.read(spec["size"] + 1)
            if len(data) != spec["size"]:
                raise ValueError("Font size mismatch")
            path.write_bytes(data)
        if not path.is_file() or digest(path) != spec["sha256"]:
            raise ValueError(f"Missing or corrupt font {path}; use --fetch")
    texts = json.loads((SOURCE / "texts.json").read_text(encoding="utf-8"))
    rows = []
    render = []
    for code, samples in texts.items():
        script = {
            "ara": "arabic",
            "urd": "arabic",
            "fas": "arabic",
            "hin": "devanagari",
            "ben": "bengali",
            "chi_sim": "chinese",
            "chi_tra": "chinese",
            "jpn": "japanese",
            "kor": "korean",
        }.get(code, "latin")
        for i, text in enumerate(samples):
            category = ("document", "terminal", "compressed-rotated")[i]
            path = Path(code) / f"{category}.{'jpg' if i == 2 else 'png'}"
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            render.append(
                {
                    "text": text,
                    "font": str((assets / fonts[script]["file"]).resolve()),
                    "output": str(target.with_suffix(".png").resolve()),
                    "dark": i == 1,
                    "rtl": code in {"ara", "urd", "fas"},
                }
            )
            rows.append(
                {
                    "id": f"{code}-{i}",
                    "path": path.as_posix(),
                    "language": code,
                    "category": category,
                    "topic": ("invoice", "connection", "train")[i],
                    "text": text,
                    "query": text.splitlines()[0],
                    "script": script,
                    "provenance": (
                        "Assistant-authored synthetic text; no independent language review"
                    ),
                    "license": "Apache-2.0 (fixture code/text); SIL-OFL-1.1 (rendering fonts)",
                }
            )
    with tempfile.TemporaryDirectory(prefix="lens-multilingual-render-") as directory:
        config = Path(directory) / "render.json"
        config.write_text(json.dumps({"images": render}, ensure_ascii=False), encoding="utf-8")
        renderer = subprocess.check_output(
            ["node", str(Path(__file__).with_name("render_multilingual.mjs")), str(config)],
            timeout=180,
            text=True,
        )
    (output / "renderer.json").write_text(renderer, encoding="utf-8")
    for row in rows:
        target = output / row["path"]
        if row["category"] == "compressed-rotated":
            with Image.open(target.with_suffix(".png")) as image:
                altered = image.filter(ImageFilter.GaussianBlur(0.4)).rotate(1, fillcolor="white")
                altered.save(target, quality=65)
                altered.close()
            target.with_suffix(".png").unlink()
        row["sha256"] = digest(target)
    (output / "manifest.json").write_text(
        json.dumps(
            {"version": 1, "seed": 20261026, "fonts": fonts, "images": rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    print(f"Built {len(build(args.output, args.assets, fetch=args.fetch))} diagnostic images")


if __name__ == "__main__":
    main()
