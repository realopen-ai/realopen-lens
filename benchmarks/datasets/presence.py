"""Build frozen, model-independent text-presence additions to the retrieval corpus."""

import argparse
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont
from benchmarks.datasets.downloads import download
from benchmarks.datasets.references import CORRECTIONS, inputs
from realopen_lens.models.manager import digest
from benchmarks.paths import SOURCES

SEED = 20261020


def build(root: Path, original: Path, cache: Path) -> dict:
    """Preserve reviewed labels; never inspect OCR or detector output during construction."""
    root.mkdir(parents=True, exist_ok=True)
    rows = inputs(original)
    for row in rows:
        target = root / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / row["path"], target)
        row["corpus"] = "historical"
    old_hashes = {r["sha256"] for r in rows}
    review = json.loads((SOURCES / "presence-photos-v1.json").read_text())
    for photo in review["photos"]:
        if photo["text_presence"] != "absent":
            continue
        source = photo["source"]
        cached = download(source["url"], cache / photo["filename"])
        if digest(cached) != photo["sha256"] or photo["sha256"] in old_hashes:
            raise ValueError("Photo checksum mismatch or overlap with historical corpus")
        relative = Path("fresh/photos") / photo["filename"]
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, root / relative)
        with Image.open(cached) as photo_image:
            width, height = photo_image.size
        rows.append(
            {
                "id": relative.as_posix(),
                "path": relative.as_posix(),
                "width": width,
                "height": height,
                "format": "JPEG",
                "frames": 1,
                "category": "photos",
                "subcategory": "reviewed-no-readable-text",
                "sha256": photo["sha256"],
                "source": source,
                "query": photo["query"],
                "language": "en",
                "split": "holdout",
                "difficulty": ["real-negative", "fresh-evaluation"],
                "expected": {"decode": "success"},
                "negative_text": True,
                "annotation": photo["review_note"],
                "corpus": "fresh",
                "family": photo["filename"],
            }
        )
    font_spec = json.loads((SOURCES / "font.json").read_text())
    font_path = download(font_spec["url"], cache / "NotoSans.ttf", font_spec["git_blob_sha1"])
    layouts = ["terminal", "editor", "dashboard", "dialog", "browser", "mobile"]
    conditions = ["small", "faint", "dark-small", "tiny-high-resolution", "faint-dark", "edge-text"]
    for layout_number, layout in enumerate(layouts):
        for condition_number, condition in enumerate(conditions):
            number = layout_number * len(conditions) + condition_number
            identifier = f"LENS-{7300 + number}"
            job = ["inventory", "backup", "billing", "telemetry", "archive", "delivery"][
                layout_number
            ]
            dark = condition in ("dark-small", "faint-dark")
            faint = condition in ("faint", "faint-dark")
            width, height = (2400, 1350) if condition == "tiny-high-resolution" else (1600, 900)
            font_size = 10 if condition in ("small", "tiny-high-resolution") else 12
            visible = [
                f"{layout.title()} - {job} service",
                f"Request {identifier}",
                f"{job}_sync failed: retry after 3500 ms",
                f"Endpoint /api/v2/{job}/{number + 1}",
                "Status: connection timeout",
            ]
            background = "#20252b" if dark else "#f6f7f8"
            ink = (
                ("#777c82" if faint else "#eeeeee") if dark else ("#b1b6bb" if faint else "#18222d")
            )
            image = Image.new("RGB", (width, height), background)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 40), fill="#323943" if dark else "#dfe4ea")
            font = ImageFont.truetype(
                str(font_path), font_size, layout_engine=ImageFont.Layout.BASIC
            )
            x, y = (width - 390, height - 140) if condition == "edge-text" else (48, 110)
            if layout == "dialog":
                draw.rectangle((x - 12, y - 15, x + 440, y + 140), outline="#808890", width=1)
            elif layout in ("editor", "browser", "dashboard"):
                draw.line((24, 70, 24, height - 30), fill="#9ba4af", width=1)
            elif layout == "mobile":
                draw.rounded_rectangle(
                    (x - 14, 65, x + 400, height - 50), radius=18, outline="#87939e"
                )
            for n, text in enumerate(visible):
                draw.text((x, y + n * 23), text, font=font, fill=ink)
            relative = Path("fresh/screenshots") / f"{layout}-{condition}.png"
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            image.save(root / relative)
            rows.append(
                {
                    "id": relative.as_posix(),
                    "path": relative.as_posix(),
                    "width": width,
                    "height": height,
                    "format": "PNG",
                    "frames": 1,
                    "category": "screenshots",
                    "subcategory": layout,
                    "sha256": digest(root / relative),
                    "source": {
                        "kind": "synthetic",
                        "license": "CC0-1.0",
                        "seed": SEED,
                        "font": font_spec,
                        "font_sha256": digest(font_path),
                    },
                    "truth": "\n".join(visible),
                    "identifiers": [identifier, f"{job}_sync", f"/api/v2/{job}/{number + 1}"],
                    "query": (
                        f"The {layout} with {job} service request {identifier} "
                        f"and connection timeout"
                    ),
                    "identifier_query": identifier,
                    "language": "en",
                    "split": "holdout",
                    "difficulty": [condition, "fresh-evaluation"],
                    "expected": {"decode": "success"},
                    "corpus": "fresh",
                    "family": layout,
                    "annotation": (
                        "Authored visible strings frozen before OCR; not a real "
                        "application screenshot."
                    ),
                }
            )
            image.close()
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate corpus images")
    manifest = {
        "version": 3,
        "seed": SEED,
        "images": rows,
        "excluded_queries": [
            r["id"] for r in json.loads(CORRECTIONS.read_text())["exclude_queries"]
        ],
        "annotation_protocol": {
            "photos": (
                "One assistant visual review before inference; no "
                "independent human/native review."
            ),
            "screenshots": (
                "Authored ground truth, six layout families; no tuning " "after evaluation."
            ),
            "fresh": "New to these runs, not a blind independently collected test set.",
        },
    }
    path = root / "manifest.json"
    content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("Dataset identity changed; use a new output directory")
    path.write_text(content)
    print(f"Built {len(rows)} unique images ({sum(r['corpus'] == 'fresh' for r in rows)} fresh)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/presence-retrieval-v1"))
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmark-500-v2"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/presence-retrieval"))
    args = parser.parse_args()
    build(args.output, args.dataset, args.cache)


if __name__ == "__main__":
    main()
