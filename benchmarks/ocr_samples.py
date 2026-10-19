"""Prepare a small, sourced terminal/document/diagram OCR diagnostic corpus."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from benchmarks.samples import save_json

SPECS = {
    "terminal/traceback.png": {
        "queries": [
            "Terminal traceback with divide_all and show_locals=True",
            "Trace Python avec divide_all et show_locals=True",
            "تتبع خطأ بايثون يتضمن divide_all و show_locals=True",
        ],
        "key_phrases": ["ZeroDivisionError", "divide_all", "show_locals", "extra_lines"],
    },
    "terminal/logging.png": {
        "queries": [
            "Server log: unable to find pomelo in database, admin request 401",
            "Journal serveur : pomelo introuvable dans la base, requête admin 401",
            "سجل خادم لا يجد pomelo في قاعدة البيانات وطلب admin يعيد 401",
        ],
        "key_phrases": ["confirmFruitPurchase", "pomelo", "/favicon.ico", "401", "127.0.0.1:8080"],
    },
    "terminal/log.png": {
        "queries": [
            "Terminal JSONRPC notify_hello with Deadpool and Rise of the Skywalker",
            "Terminal JSONRPC notify_hello avec Deadpool et Rise of the Skywalker",
            "طرفية تعرض JSONRPC و notify_hello وفيلمي Deadpool و Rise of the Skywalker",
        ],
        "key_phrases": ["notify_hello", "Deadpool", "Skywalker", "jsonrpc"],
    },
    "document/phototest.tif": {
        "queries": [
            "OCR document with 12 point text and the quick brown dog jumping over the lazy fox",
            "Document OCR avec du texte de 12 points et un chien sautant au-dessus du renard",
            "وثيقة اختبار التعرف على النص بحجم 12 نقطة وكلب يقفز فوق ثعلب كسول",
        ],
        "key_phrases": ["12 point", "file format", "lazy fox"],
    },
    "document/8071_093.3B.tif": {
        "queries": [
            "Scanned article What Kids Need Most in a Dad with Kathleen Kilpatrick",
            "Article numérisé sur les besoins des enfants envers leur père et Kathleen Kilpatrick",
            "مقال ممسوح عن احتياجات الأطفال من الأب يذكر Kathleen Kilpatrick",
        ],
        "key_phrases": ["Kilpatrick", "Guarendi", "CIGNA", "fatherhood"],
    },
    "document/8087_054.3B.tif": {
        "queries": [
            "Scanned travel article about Dubrovnik, the Dalmatian coast and tourism",
            "Article de voyage numérisé sur Dubrovnik, la côte dalmate et le tourisme",
            "مقال سفر ممسوح عن Dubrovnik وساحل دالماسيا والسياحة",
        ],
        "key_phrases": ["Dubrovnik", "Dalmatian", "tourism", "Adriatic"],
    },
    "diagram/SystemContext.png": {
        "queries": [
            "System Context View diagram for Internet Banking System",
            "Schéma de contexte du système de banque en ligne",
            "مخطط سياق نظام الخدمات المصرفية عبر الإنترنت",
        ],
        "key_phrases": ["System Context View", "Simple Email Service", "Core Banking System"],
    },
    "diagram/Containers.png": {
        "queries": [
            "Container diagram with Static Content, UI, Backend, Spring Boot and MySQL",
            "Schéma de conteneurs avec Static Content, UI, Backend, Spring Boot et MySQL",
            "مخطط حاويات يتضمن Static Content و UI و Backend و Spring Boot و MySQL",
        ],
        "key_phrases": ["Static Content", "Spring Boot", "MySQL", "Statement Store"],
    },
    "diagram/Components.png": {
        "queries": [
            "Component diagram with Accounts Summary API, Sign In API and Security Component",
            "Schéma de composants avec Accounts Summary API, Sign In API et Security Component",
            "مخطط مكونات يتضمن Accounts Summary API و Sign In API و Security Component",
        ],
        "key_phrases": ["Sign In API", "Accounts Summary API", "Security Component", "Spring MVC"],
    },
}


def manifest() -> list[dict[str, Any]]:
    return json.loads(files("benchmarks").joinpath("ocr_assets/manifest.json").read_text())


def ensure_download(url: str, target: Path, expected: str) -> None:
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Existing file differs from pinned asset: {target}")
        return
    executable = shutil.which("curl")
    if not executable:
        raise ValueError("The optional OCR sample downloader requires curl on PATH")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        temporary = Path(directory) / "download"
        subprocess.run(
            [
                executable,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                "60",
                "--max-filesize",
                "10485760",
                url,
                "--output",
                str(temporary),
            ],
            check=True,
        )
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Downloaded asset checksum mismatch: {url}")
        temporary.replace(target)


def publish(path: Path, value: Any) -> None:
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"Refusing to overwrite edited annotations: {path}")
    save_json(path, value)


def prepare(output: Path, include_photos: bool = False) -> None:
    entries = manifest()
    truth, queries = [], []
    for item in entries:
        image_id = item["image_id"]
        ensure_download(item["url"], output / image_id, item["sha256"])
        if "reference_url" in item:
            reference = output / "references" / (Path(image_id).stem + ".txt")
            ensure_download(item["reference_url"], reference, item["reference_sha256"])
            transcription = reference.read_text(encoding="utf-8")
        else:
            transcription = (
                files("benchmarks")
                .joinpath("ocr_assets/" + item["reference_resource"])
                .read_text(encoding="utf-8")
            )
        truth.append(
            {
                "image_id": image_id,
                "language": item["language"],
                "category": item["category"],
                "text": transcription,
                "key_phrases": SPECS[image_id]["key_phrases"],
                "reference_source": item.get("reference_url", item["source"]),
                "reading_order": (
                    "Upstream transcript, or node blocks then relationships for diagrams"
                ),
            }
        )
        for language, text in zip(("en", "fr", "ar"), SPECS[image_id]["queries"], strict=True):
            queries.append(
                {
                    "id": f"{Path(image_id).stem}-{language}",
                    "text": text,
                    "language": language,
                    "category": item["category"],
                    "relevant": [image_id],
                    "judgment_source": (
                        "Assistant-authored after visual inspection; "
                        "translations not independently reviewed"
                    ),
                }
            )
    publish(output / "manifest.json", entries)
    publish(output / "ocr-truth.json", truth)
    publish(output / "queries.json", queries)
    if include_photos:
        root = output.parent
        photos = root / "public-sample"
        photo_queries = json.loads((photos / "queries.caption-derived.json").read_text())
        relative = output.relative_to(root).as_posix()
        mixed_truth = [{**item, "image_id": relative + "/" + item["image_id"]} for item in truth]
        mixed_queries = [
            {**item, "relevant": [relative + "/" + p for p in item["relevant"]]} for item in queries
        ]
        mixed_queries += [
            {
                **item,
                "category": "photo",
                "relevant": ["public-sample/" + p for p in item["relevant"]],
            }
            for item in photo_queries
        ]
        publish(root / "ocr-truth.mixed.json", mixed_truth)
        publish(root / "queries.mixed.json", mixed_queries)
    print(
        f"Prepared {len(entries)} sourced text-heavy images and {len(queries)} queries in {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ocr-sample"))
    parser.add_argument(
        "--include-photos",
        action="store_true",
        help="Also prepare mixed queries using sibling public-sample/ photographs",
    )
    args = parser.parse_args()
    try:
        prepare(args.output, args.include_photos)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
