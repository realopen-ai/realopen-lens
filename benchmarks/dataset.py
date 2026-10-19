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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from benchmarks.samples import annotations, save_json
from benchmarks.coco_licenses import load as coco_licenses
from benchmarks.ocr_samples import prepare as prepare_ocr, SPECS

SEED = 20261019
SOURCES = Path(__file__).parent / "sources"
COCO_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
FORMS = (
    "f1040 fw9 fw4 f1099msc f1099nec f941 f940 f1120 "
    "f1065 f4506t f8822 fss4 f2553 f2848 f4868 f8962"
).split()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, path: Path, blob: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "2",
                "--max-time",
                "90",
                "--max-filesize",
                "350000000",
                url,
                "-o",
                str(part),
            ],
            check=True,
            capture_output=True,
        )
        if blob:
            payload = part.read_bytes()
            actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if actual != blob:
                raise ValueError(f"Git object integrity mismatch: {url}")
        part.replace(path)
    if blob:
        payload = path.read_bytes()
        if hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() != blob:
            raise ValueError(f"Cached Git object integrity mismatch: {path}")
    return path


def font(size: int, path: Path) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.BASIC)


def family_split(family: str) -> str:
    return "holdout" if int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 5 == 0 else "dev"


def metadata(
    path: Path,
    root: Path,
    category: str,
    subcategory: str,
    source: dict,
    family: str | None = None,
    **extra: Any,
) -> dict:
    relative = path.relative_to(root).as_posix()
    with Image.open(path) as image:
        width, height = image.size
        mode, fmt, frames = image.mode, image.format, getattr(image, "n_frames", 1)
        gray = image.convert("L").resize((9, 8))
        values = np.asarray(gray)
        dhash = sum(int(v) << i for i, v in enumerate((values[:, 1:] > values[:, :-1]).flat))
    family = family or relative
    return {
        "id": relative,
        "path": relative,
        "category": category,
        "subcategory": subcategory,
        "width": width,
        "height": height,
        "format": fmt,
        "mode": mode,
        "frames": frames,
        "sha256": digest(path),
        "dhash": f"{dhash:016x}",
        "source": source,
        "family": family,
        "split": family_split(family),
        "difficulty": [],
        "expected": {"decode": "reject" if width * height > 24_000_000 else "success"},
        **extra,
    }


def synthetic(root: Path, font_path: Path) -> list[dict]:
    """Distinct authored content and layouts; synthetic quality is reported separately."""
    rows: list[dict[str, Any]] = []
    rng = random.Random(SEED)
    topics = [
        "Orion",
        "Cedar",
        "Atlas",
        "Harbor",
        "Maple",
        "Aurora",
        "Falcon",
        "Nimbus",
        "Willow",
        "Summit",
        "Coral",
        "Quartz",
        "Juniper",
        "Lunar",
        "Delta",
        "Violet",
        "Sierra",
        "Mosaic",
        "Aspen",
        "Solstice",
    ]
    for category, count in [("documents", 80), ("screenshots", 20), ("diagrams", 20)]:
        for i in range(count):
            lang = "fr" if i % 2 else "en"
            style = (
                ["invoice", "receipt", "form", "report"][i // 20]
                if category == "documents"
                else (
                    ["conversation", "mobile", "dashboard", "terminal"][i // 5]
                    if category == "screenshots"
                    else ["flowchart", "architecture", "graph", "drawing"][i // 5]
                )
            )
            width, height = (
                (700, 1250)
                if style in ("mobile", "receipt")
                else (
                    (1600, 1000)
                    if category == "screenshots"
                    else (1200, 1600) if category == "documents" else (1400, 900)
                )
            )
            if style == "report":
                height = 2100
            background = "#17222d" if style == "terminal" else "#ffffff"
            foreground = "#e3eaf2" if style == "terminal" else "#182c40"
            image = Image.new("RGB", (width, height), background)
            draw = ImageDraw.Draw(image)
            code = f"LENS-{category[:3].upper()}-{i:04d}"
            topic = topics[(i * 7) % len(topics)]
            title = {
                "invoice": "FACTURE" if lang == "fr" else "INVOICE",
                "receipt": "REÇU" if lang == "fr" else "RECEIPT",
                "form": "FORMULAIRE" if lang == "fr" else "APPLICATION FORM",
                "report": "RAPPORT" if lang == "fr" else "OPERATIONS REPORT",
                "conversation": "Équipe produit" if lang == "fr" else "Product team",
                "mobile": "Suivi de livraison" if lang == "fr" else "Delivery tracking",
                "dashboard": "Tableau de bord" if lang == "fr" else "Service dashboard",
                "terminal": "Deployment console",
                "flowchart": "Order workflow",
                "architecture": "Service architecture",
                "graph": "Monthly energy usage",
                "drawing": "Sensor assembly drawing",
            }[style]
            lines = [f"{title} {topic}", code]
            if category == "documents":
                lines += (
                    [
                        "Client : Atelier " + topic,
                        f"Date : 2026-09-{i % 28 + 1:02d}",
                        "Adresse : 42 rue des Jardins",
                        "Référence : commande confirmée",
                        f"Quantité : {i % 9 + 1}",
                        f"Total : {127 + i * 13},50 EUR",
                        "Livraison : sous cinq jours ouvrés",
                        "Paiement : virement bancaire",
                    ]
                    if lang == "fr"
                    else [
                        "Customer: " + topic + " Workshop",
                        f"Date: 2026-09-{i % 28 + 1:02d}",
                        "Address: 42 Garden Street",
                        "Reference: confirmed order",
                        f"Quantity: {i % 9 + 1}",
                        f"Total: {127 + i * 13}.50 EUR",
                        "Delivery: within five business days",
                        "Payment: bank transfer",
                    ]
                )
                if style == "report":
                    lines += [
                        f"Section {j}: {topic} capacity {rng.randrange(10, 99)} percent"
                        for j in range(1, 10)
                    ]
            elif category == "screenshots":
                lines += (
                    [
                        "Bonjour, le serveur est-il disponible ?",
                        "Oui, la migration est terminée.",
                        f"Ticket : SRV_{i:04d}",
                        "Prochaine étape : vérifier les journaux.",
                    ]
                    if lang == "fr"
                    else [
                        "Is the server available?",
                        "Yes, the migration is complete.",
                        f"Ticket: SRV_{i:04d}",
                        "Next step: check the deployment logs.",
                    ]
                )
                if style == "terminal":
                    lines += [
                        f"deploy_{topic.lower()} --port {8000+i}",
                        "ConnectionError: database connection refused",
                        "Retry scheduled in 30s",
                    ]
            elif style == "graph":
                lines += ["January", "February", "March", "April", f"Total: {230+i*5} kWh"]
            elif style == "drawing":
                lines += [
                    "Mounting bracket",
                    f"Width: {80+i} mm",
                    f"Height: {40+i} mm",
                    "Tolerance: 0.2 mm",
                ]
            else:
                lines += [
                    "Input queue",
                    "Validation service",
                    f"{topic} processor",
                    "Database",
                    "Monitoring",
                    f"Capacity: {50+i*7} requests per second",
                ]
            if style == "mobile":
                lines = lines[:2] + (
                    [
                        f"Colis : PKG_{i:04d}",
                        "En cours de livraison",
                        "Arrivée prévue : 16:30",
                        "Destinataire : Atelier " + topic,
                    ]
                    if lang == "fr"
                    else [
                        f"Package: PKG_{i:04d}",
                        "Out for delivery",
                        "Estimated arrival: 16:30",
                        "Recipient: " + topic + " Workshop",
                    ]
                )
            text_size = 26 if style == "receipt" else 30
            y = 50
            for j, line in enumerate(lines):
                size = text_size + 8 if j == 0 else text_size
                if style == "graph" and j >= 2:
                    if j < 6:
                        x = 150 + (j - 2) * 280
                        top = 740 - (100 + rng.randrange(100, 350))
                        draw.rectangle(
                            (x, top, x + 140, 740),
                            fill=["#237ba0", "#e6a23c", "#78a55a", "#9a70b4"][j - 2],
                        )
                        draw.text((x - 15, 760), line, font=font(24, font_path), fill=foreground)
                    else:
                        draw.text((80, 220), line, font=font(28, font_path), fill=foreground)
                    draw.line((100, 320, 100, 740, 1300, 740), fill=foreground, width=3)
                elif style == "drawing" and j >= 2:
                    draw.text(
                        (70, 200 + (j - 2) * 55), line, font=font(26, font_path), fill=foreground
                    )
                    draw.rectangle((250, 490, 1150, 760), outline=foreground, width=5)
                    for cx in (350, 1050):
                        draw.ellipse((cx - 50, 575, cx + 50, 675), outline=foreground, width=4)
                    draw.line((250, 790, 1150, 790), fill=foreground, width=2)
                    for cx in (250, 1150):
                        draw.line((cx, 770, cx, 810), fill=foreground, width=2)
                elif category == "diagrams" and j >= 2:
                    y = 190 + ((j - 2) // 2) * 170
                    x = 50 + ((j - 2) % 2) * 660
                    draw.rectangle((x, y, x + 610, y + 100), outline="#1b6fa0", width=3)
                    draw.text((x + 15, y + 30), line, font=font(24, font_path), fill=foreground)
                    if j % 2 == 0:
                        draw.line((x + 610, y + 50, x + 650, y + 50), fill="#1b6fa0", width=3)
                else:
                    if style in ("mobile", "conversation") and j > 1:
                        draw.rounded_rectangle(
                            (35, y - 10, width - 35, y + 55),
                            radius=18,
                            fill="#e5f1fb" if j % 2 else "#eef3ed",
                        )
                    if style in ("form", "invoice", "dashboard") and j > 1:
                        draw.rectangle((35, y - 10, width - 35, y + 55), outline="#b9c5ce", width=2)
                    draw.text((50, y), line, font=font(size, font_path), fill=foreground)
                    y += 80 if style != "receipt" else 90
            extension = [".png", ".jpg", ".webp", ".tif", ".bmp"][i % 5]
            path = root / category / "synthetic" / f"{style}-{i:03d}{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            source = {
                "kind": "synthetic",
                "generator": "benchmarks.dataset.synthetic/v2",
                "seed": SEED,
                "license": "CC0-1.0",
                "topic": topic,
            }
            row = metadata(
                path,
                root,
                category,
                style,
                source,
                f"synthetic/{style}",
                language=lang,
                truth="\n".join(lines),
                key_phrases=[code, topic],
                query=f"{title} {topic}",
                identifier_query=code,
                difficulty=["synthetic"],
            )
            rows.append(row)
    return rows


def annotation_matches(row: dict, annotation: dict) -> bool:
    """A basename is not a source identity (e.g. unrelated header.png images)."""
    source = row["source"]
    return (
        source.get("repo") == annotation["repo"] and source.get("path") == annotation["source_path"]
    )


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
