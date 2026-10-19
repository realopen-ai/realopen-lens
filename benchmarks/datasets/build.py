"""Build a pinned, diverse local dataset. Downloads are explicit; inference stays offline."""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote
import numpy as np
from PIL import Image, ImageFilter
from benchmarks.datasets.photos import annotations, save_json
from benchmarks.datasets.licenses import load as coco_licenses
from benchmarks.datasets.ocr import prepare as prepare_ocr, SPECS
from benchmarks.datasets.downloads import digest, download
from benchmarks.datasets.metadata import family_split, metadata, annotation_matches
from benchmarks.datasets.synthetic import synthetic
from benchmarks.datasets.specs import SEED, FORMS
from benchmarks.paths import SOURCES


def build(root: Path, cache: Path, workers: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "manifest.json"
    if lock.exists():
        manifest = json.loads(lock.read_text())
        for row in manifest["images"]:
            if digest(root / row["path"]) != row["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {row['id']}")
        print(f"Verified {len(manifest['images'])} existing cases")
        return
    font_spec = json.loads((SOURCES / "font.json").read_text())
    font_path = download(font_spec["url"], cache / "NotoSans.ttf", font_spec["git_blob_sha1"])
    catalog = json.loads((SOURCES / "catalog.json").read_text())
    rows: list[dict[str, Any]] = []

    def remote(spec: dict) -> dict:
        url = (
            f"https://raw.githubusercontent.com/{spec['repo']}/"
            f"{spec['commit']}/{quote(spec['path'])}"
        )
        short = hashlib.sha256(spec["path"].encode()).hexdigest()[:10]
        path = (
            root
            / spec["category"]
            / spec["repo"].replace("/", "-")
            / (short + "-" + Path(spec["path"]).name)
        )
        download(url, path, spec["git_blob_sha1"])
        return metadata(
            path, root, spec["category"], spec["subcategory"], {**spec, "url": url, "kind": "real"}
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows.extend(pool.map(remote, catalog["sources"]))
    unique = []
    seen_hashes = set()
    counts: dict[str, int] = {}
    for row in rows:
        limit = {"screenshots": 100, "diagrams": 57, "documents": 4}[row["category"]]
        if row["sha256"] in seen_hashes or counts.get(row["category"], 0) >= limit:
            continue
        unique.append(row)
        seen_hashes.add(row["sha256"])
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    rows = unique
    if counts != {"screenshots": 100, "diagrams": 57, "documents": 4}:
        raise ValueError(f"Insufficient unique source images: {counts}")
    print(f"Ready: {len(rows)} pinned, deduplicated repository images", flush=True)
    # Official per-image COCO licenses, filtered to attribution/share-alike/public-domain.
    info = coco_licenses(cache)
    allowed = {x["id"]: x for x in info["licenses"] if x["id"] in (4, 5, 7, 8)}
    image_info = {x["id"]: x for x in info["images"] if x["license"] in allowed}
    captions = annotations(Path(".cache/sample-downloads"))
    candidates = sorted(
        [x for x in captions if x.get("split") == "test" and x.get("cocoid") in image_info],
        key=lambda x: x["filename"],
    )
    selected = random.Random(SEED).sample(candidates, 150)

    def photo(item: dict) -> dict:
        url = "https://s3.amazonaws.com/images.cocodataset.org/val2014/" + item["filename"]
        path = root / "photos" / item["filename"]
        download(url, path)
        record = image_info[item["cocoid"]]
        source = {
            "kind": "real",
            "url": url,
            "dataset": "COCO2014/Karpathy test",
            "coco_id": item["cocoid"],
            "license": allowed[record["license"]],
            "flickr_url": record.get("flickr_url"),
            "license_annotations_sha256": digest(cache / "coco-license-info.json"),
        }
        caption = item["sentences"][0]["raw"].strip()
        bucket = next(
            (
                name
                for name, words in [
                    (
                        "animals",
                        ("dog", "cat", "bird", "bear", "horse", "zebra", "giraffe", "elephant"),
                    ),
                    ("people", ("man", "woman", "people", "person", "boy", "girl")),
                    ("transport", ("bus", "train", "car", "plane", "truck")),
                    ("food", ("food", "pizza", "sandwich", "cake", "plate")),
                    ("indoor", ("room", "kitchen", "bathroom", "bed")),
                ]
                if any(w in caption.lower().split() for w in words)
            ),
            "objects-outdoor",
        )
        return metadata(
            path,
            root,
            "photos",
            bucket,
            source,
            query=caption,
            language="en",
            difficulty=["caption-derived-judgment"],
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows.extend(pool.map(photo, selected))
    print("Ready: 150 licensed photos", flush=True)
    for form in FORMS:
        pdf = cache / "pdf" / f"{form}.pdf"
        url = f"https://www.irs.gov/pub/irs-pdf/{form}.pdf"
        download(url, pdf)
        page = 2 if form in ("f1099msc", "f1099nec") else 1
        path = root / "documents" / "pdf-forms" / f"{form}-page{page}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-singlefile",
                    "-scale-to",
                    "2200",
                    "-png",
                    str(pdf),
                    str(path.with_suffix("")),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        rows.append(
            metadata(
                path,
                root,
                "documents",
                "pdf-rendered-form",
                {
                    "kind": "real",
                    "url": url,
                    "pdf_sha256": digest(pdf),
                    "license": "US federal government work / public domain",
                    "render": f"Poppler page {page}, scale-to 2200",
                },
            )
        )
    manual = json.loads((SOURCES / "manual_queries.json").read_text())
    for annotation in manual:
        for row in rows:
            if annotation_matches(row, annotation):
                row["query"] = annotation["query"]
                row["language"] = "en"
                if annotation.get("key_phrases"):
                    row["key_phrases"] = annotation["key_phrases"]
                    row["reference_kind"] = "visually-reviewed-key-phrases"
    for row in rows:
        if row["subcategory"] == "pdf-rendered-form":
            form = Path(row["path"]).name.split("-page")[0]
            page = 2 if form in ("f1099msc", "f1099nec") else 1
            pdf = cache / "pdf" / f"{form}.pdf"
            reference = json.loads((SOURCES / "pdf_references.json").read_text())[form]
            if digest(pdf) != reference["sha256"]:
                raise ValueError(f"PDF changed since reference extraction: {form}")
            row.update(
                truth=reference["text"],
                language="en",
                reference_kind="source-PDF-text-layer",
                query=f"IRS form {form[1:]} tax document",
            )
    # Previously inspected C4 examples: attribution and reference transcriptions retained.
    old_root = cache / "ocr-sample"
    prepare_ocr(old_root)
    old_truth = json.loads((old_root / "ocr-truth.json").read_text())
    for item in old_truth:
        old = old_root / item["image_id"]
        if item["category"] == "diagram":
            path = root / "diagrams" / "c4" / old.name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, path)
            rows.append(
                metadata(
                    path,
                    root,
                    "diagrams",
                    "architecture",
                    {
                        "kind": "real",
                        "url": "https://c4model.com/images/examples/" + old.name,
                        "license": "CC-BY-4.0",
                        "author": "Simon Brown",
                    },
                    truth=item["text"],
                    key_phrases=item.get("key_phrases", []),
                    language="en",
                )
            )
        else:
            matching = [r for r in rows if r["source"].get("path", "").endswith("/" + old.name)]
            for row in matching:
                row.update(
                    truth=item["text"], key_phrases=item.get("key_phrases", []), language="en"
                )
    # eurotext is a mixed-language source transcript, not an OCR-generated label.
    euro = next(r for r in rows if r["source"].get("path") == "testing/eurotext.tif")
    ref = cache / "eurotext.txt"
    download(
        "https://raw.githubusercontent.com/tesseract-ocr/test/"
        "232ff181c66516116ec0e84c4963f70de15050fd/testing/eurotext.txt",
        ref,
    )
    euro.update(truth=ref.read_text(), language="en+de+fr")
    rows.extend(synthetic(root, font_path))
    # Hold out entire sources/templates, keeping known earlier diagnostic cases in dev.
    for row in rows:
        source = row["source"]
        if source.get("repo"):
            row["family"] = "source/" + source["repo"]
        elif row["subcategory"] == "pdf-rendered-form":
            row["family"] = "source/irs"
        elif source.get("author") == "Simon Brown":
            row["family"] = "source/c4"
        row["split"] = family_split(row["family"])
        if source.get("repo") in ("Textualize/rich", "tesseract-ocr/test"):
            row["split"] = "dev"  # These sources occurred in the earlier diagnostic runs.
        if source.get("repo") == "mermaid-js/mermaid":
            row["split"] = "holdout"
        if source.get("repo") or source["kind"] == "synthetic":
            row["difficulty"].append("intentional-source-or-template-family")
    translations = json.loads((SOURCES / "photo_translations.json").read_text())
    for row in rows:
        if row["id"] in translations:
            row["query_variants"] = [
                {"language": lang, "text": text} for lang, text in translations[row["id"]].items()
            ]
        for key, spec in SPECS.items():
            if Path(row["path"]).name.endswith(Path(key).name):
                row["query_variants"] = [
                    {"language": lang, "text": text}
                    for lang, text in zip(("en", "fr", "ar"), spec["queries"], strict=True)
                ]
    # Intentionally related variants stay in their parent's split/relevance family.
    rng = random.Random(SEED)
    parents = []
    for cat, n in [("photos", 15), ("screenshots", 15), ("documents", 10), ("diagrams", 10)]:
        parents.extend(rng.sample([r for r in rows if r["category"] == cat], n))
    transforms = [
        "low-resolution",
        "blur",
        "compression",
        "noise",
        "rotated-90",
        "crop",
        "high-resolution",
        "rgba",
        "palette",
        "multipage",
    ]
    for i, parent in enumerate(parents):
        transform = transforms[i % len(transforms)]
        with Image.open(root / parent["path"]) as source:
            im = source.convert("RGB")
        extension = ".png"
        if transform == "low-resolution":
            im.thumbnail((320, 240))
        elif transform == "blur":
            im = im.filter(ImageFilter.GaussianBlur(2.5))
        elif transform == "compression":
            extension = ".jpg"
        elif transform == "noise":
            a = np.asarray(im, dtype=np.int16)
            noise = np.random.default_rng(SEED + i).normal(0, 22, a.shape)
            im = Image.fromarray(np.clip(a + noise, 0, 255).astype(np.uint8))
        elif transform == "rotated-90":
            im = im.transpose(Image.Transpose.ROTATE_90)
        elif transform == "crop":
            im = im.crop((im.width // 4, im.height // 4, im.width * 3 // 4, im.height * 3 // 4))
        elif transform == "high-resolution":
            im = im.resize((5500, 4500) if i < 30 else (4000, 3000))
        elif transform == "rgba":
            canvas = Image.new("RGBA", (im.width + 80, im.height + 80), (0, 0, 0, 0))
            canvas.paste(im, (40, 40))
            im = canvas
        elif transform == "palette":
            im = im.quantize(colors=16)
        elif transform == "multipage":
            extension = ".tif"
        path = root / "challenging" / transform / f"{i:03d}{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        options = (
            {"quality": 12}
            if transform == "compression"
            else (
                {"save_all": True, "append_images": [Image.new("RGB", (100, 100), "red")]}
                if transform == "multipage"
                else {}
            )
        )
        im.save(path, **options)
        extra = {
            k: parent[k]
            for k in ("truth", "key_phrases", "language", "query")
            if k in parent and transform != "crop"
        }
        rows.append(
            metadata(
                path,
                root,
                "challenging",
                transform,
                {
                    "kind": "derived",
                    "parent": parent["id"],
                    "parent_sha256": parent["sha256"],
                    "license": parent["source"]["license"],
                    "seed": SEED + i,
                },
                parent["family"],
                difficulty=[transform, "intentional-related-variant"],
                **extra,
            )
        )
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        if row["source"]["kind"] == "derived":
            row["split"] = by_id[row["source"]["parent"]]["split"]
    hashes: dict[str, str] = {}
    for row in rows:
        if row["sha256"] in hashes:
            raise ValueError(f"Exact duplicate: {row['id']} and {hashes[row['sha256']]}")
        hashes[row["sha256"]] = row["id"]
    near = []
    for i, left in enumerate(rows):
        for right in rows[:i]:
            distance = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
            if distance <= 3:
                near.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "distance": distance,
                        "same_family": left["family"] == right["family"],
                    }
                )
    save_json(root / "near-duplicates.json", near)
    # Keep reserve downloads and rejected duplicate sources outside the frozen collection.
    included = {row["path"] for row in rows}
    for path in list(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".webp",
            ".bmp",
        }:
            if path.relative_to(root).as_posix() not in included:
                excluded = cache / "excluded" / path.relative_to(root)
                excluded.parent.mkdir(parents=True, exist_ok=True)
                path.replace(excluded)
    rows.sort(key=lambda r: r["id"])
    save_json(
        lock,
        {
            "version": 3,
            "seed": SEED,
            "images": rows,
            "source_catalog_sha256": digest(SOURCES / "catalog.json"),
            "manual_annotations_sha256": digest(SOURCES / "manual_queries.json"),
            "font": {**font_spec, "sha256": digest(font_path)},
            "limitations": [
                "OCR truth is available only for authored or upstream-transcribed images",
                "COCO captions supply known positives, not exhaustive relevance",
                "Near-duplicate dHash flags require visual review; "
                "related variants are intentional",
                "Actual Arabic OCR represented only by one upstream test image; "
                "no Arabic transcript available",
            ],
        },
    )
    print(f"Dataset ready: {len(rows)} images; manifest: {lock}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/benchmark-500-v3"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/hardening"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    build(args.output, args.cache, args.workers)


if __name__ == "__main__":
    main()
