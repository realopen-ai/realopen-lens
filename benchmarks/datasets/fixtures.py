"""Small generated text-image fixtures, explicitly unsuitable for quality claims."""

import json
import os
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw, ImageFont, features

TOPICS = {
    "database": {
        "en": (
            "Database connection error",
            "Docker cannot connect to PostgreSQL.",
            "Connection refused on port 5432.",
        ),
        "fr": (
            "Erreur de connexion",
            "Docker ne peut pas joindre PostgreSQL.",
            "Connexion refusée sur le port 5432.",
        ),
        "ar": ("خطأ في الاتصال بقاعدة البيانات", "تعذر الاتصال بخادم PostgreSQL", "تم رفض الاتصال"),
    },
    "invoice": {
        "en": ("Invoice", "Payment for a laptop computer.", "Total: 950 EUR"),
        "fr": ("Facture", "Paiement pour un ordinateur portable.", "Total : 950 EUR"),
        "ar": ("فاتورة", "دفع ثمن حاسوب محمول", "المجموع 950 يورو"),
    },
    "meeting": {
        "en": ("Meeting notes", "Discuss the new website design.", "Next meeting: Monday."),
        "fr": (
            "Compte rendu",
            "Discussion sur la conception du site web.",
            "Prochaine réunion : lundi.",
        ),
        "ar": ("ملاحظات الاجتماع", "مناقشة تصميم الموقع الجديد", "الاجتماع القادم يوم الاثنين"),
    },
}


def create(output: Path, font: Path | None) -> None:
    if output.exists():
        raise ValueError(f"Fixture directory already exists: {output}")
    has_raqm = features.check_feature("raqm")
    candidates = (
        [str(font)]
        if font
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "DejaVuSans.ttf",
            str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"),
        ]
    )
    selected = None
    for candidate in candidates:
        try:
            selected = ImageFont.truetype(candidate, 32)
            break
        except OSError:
            continue
    if selected is None:
        raise ValueError("Pass --font with a TrueType font supporting Latin and Arabic")
    output.mkdir(parents=True)
    query_items = []
    truth = []
    for topic, translations in TOPICS.items():
        translations = {
            key: value for key, value in translations.items() if key != "ar" or has_raqm
        }
        for language, lines in translations.items():
            image = Image.new("RGB", (1000, 500), "white")
            draw = ImageDraw.Draw(image)
            rtl = language == "ar"
            for i, line in enumerate(lines):
                layout: dict[str, Any] = {"direction": "rtl" if rtl else "ltr"} if has_raqm else {}
                draw.text(
                    (950 if rtl else 50, 60 + i * 100),
                    line,
                    font=selected,
                    fill="black",
                    anchor="ra" if rtl else "la",
                    **layout,
                )
            image.save(output / f"{topic}-{language}.png")
            truth.append(
                {
                    "image_id": f"{topic}-{language}.png",
                    "text": " ".join(lines),
                    "language": language,
                }
            )
            query_items.append(
                {
                    "id": f"{topic}-{language}",
                    "text": " ".join(lines[:2]),
                    "language": language,
                    "relevant": [f"{topic}-{lang}.png" for lang in translations],
                }
            )
    (output / "queries.json").write_text(
        json.dumps(query_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "fixture-info.json").write_text(
        json.dumps(
            {
                "kind": "generated-smoke",
                "arabic_rendered": has_raqm,
                "note": "Arabic fixtures omitted when Pillow lacks RAQM shaping support.",
            }
        ),
        encoding="utf-8",
    )
    (output / "ocr-truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
