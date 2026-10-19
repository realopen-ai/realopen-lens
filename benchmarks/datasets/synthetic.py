"""Deterministic authored image layouts for benchmark datasets."""

import random
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw
from benchmarks.datasets.metadata import font, metadata
from benchmarks.datasets.specs import SEED


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
