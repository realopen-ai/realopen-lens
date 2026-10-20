"""Freeze multilingual text and stratified image inputs before encoder comparisons."""

import argparse
from collections import defaultdict
import random
from pathlib import Path

from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from realopen_lens.encoders.onnx import image_rgb
from realopen_lens.models.manager import artifact_path, digest
from realopen_lens.storage.artifacts import load_manifest
from realopen_lens.storage.catalog import Catalog
from tokenizers import Tokenizer


def prepare(
    index_dir: Path,
    collection: str,
    corpus: Path,
    judgments: Path,
    models: Path,
    output: Path,
    seed: int = 20261020,
    full_corpus: bool = False,
) -> dict:
    manifest, bundle = load(corpus, judgments)
    if output.exists():
        raise ValueError("Probe inputs already exist; preserve their frozen identity")
    catalog = Catalog(index_dir, readonly=True)
    try:
        found, generation = catalog.snapshot(collection)
        saved = load_manifest(catalog, generation)
    finally:
        catalog.close()
    groups: dict[str, list] = defaultdict(list)
    for row in manifest["images"]:
        groups[row["category"]].append(row)
    rng = random.Random(seed)
    selected = []
    for key, values in sorted(groups.items()):
        selected.extend(rng.sample(sorted(values, key=lambda r: r["sha256"]), min(6, len(values))))
    # Explicit known misses supplement, rather than replace, the seeded sample.
    target_families = {"v2-cafeteria", "v2-inkscape-welcome", "v2-wikipedia-home"}
    targets = {
        r["image_sha256"]
        for q in bundle["queries"]
        if q["family"] in target_families
        for r in q["judgments"]
        if r["grade"] > 0
    }
    selected.extend(r for r in manifest["images"] if r["sha256"] in targets)
    if full_corpus:
        selected = manifest["images"]
    selected = list({row["sha256"]: row for row in selected}.values())
    output.mkdir(parents=True)
    images = []
    for row in selected:
        source = Path(found["root"]) / row["path"]
        if digest(source) != row["sha256"]:
            raise ValueError("Probe source differs from frozen corpus")
        path = source
        if not full_corpus:
            path = output / "images" / (row["sha256"] + ".png")
            path.parent.mkdir(exist_ok=True)
            with image_rgb(source) as image:
                image.save(path)
        images.append(
            {
                "id": row["sha256"],
                "path": str(path.resolve()),
                "sha256": digest(path),
                "source_sha256": row["sha256"],
                "category": row["category"],
            }
        )
    tokenizer = Tokenizer.from_file(str(artifact_path(models, "e5", "tokenizer")))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    passages = []
    chosen = {r["id"] for r in images}
    for row in saved["ocr"]:
        if row["image_id"] not in chosen or not row["text"].strip():
            continue
        ids = tokenizer.encode(row["text"][:32000], add_special_tokens=False).ids
        # The exact production 440-token windows with 400-token stride.
        for i in list(range(0, len(ids), 400))[:16]:
            passages.append(
                {
                    "id": row["image_id"] + ":" + str(i),
                    "image_id": row["image_id"],
                    "text": tokenizer.decode(ids[i : i + 440]),
                    "language": "observed-ocr",
                }
            )
    queries = [
        {"id": q["id"], "text": q["text"], "language": q["language"]} for q in bundle["queries"]
    ]
    queries.extend(
        [
            {
                "id": "control-uppercase",
                "text": "ERR_42 FileNotFoundError /Users/Test",
                "language": "en",
            },
            {
                "id": "control-arabic-diacritics",
                "text": "مَكْتَبَةُ الْمُسْتَخْدِمِ",
                "language": "ar",
            },
            {
                "id": "control-long-arabic",
                "text": "معلومات واجهة المستخدم " * 100,
                "language": "ar",
            },
        ]
    )
    config = {
        "version": 1,
        "seed": seed,
        "full_corpus": full_corpus,
        "generation": generation["id"],
        "corpus_sha256": digest(corpus),
        "judgments_sha256": digest(judgments),
        "images": images,
        "queries": queries,
        "passages": passages,
        "limitations": [
            "Previously inspected corpus and queries; diagnostic only.",
            (
                "All original image files use production ONNX decoding."
                if full_corpus
                else (
                    "Stratified images share production EXIF/alpha decoding; "
                    "resize/normalization are compared."
                )
            ),
        ],
    }
    save_json(output / "inputs.json", config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index-dir", "corpus", "judgments", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--full-corpus", action="store_true")
    args = parser.parse_args()
    result = prepare(**vars(args))
    print({key: len(result[key]) for key in ("images", "queries", "passages")})


if __name__ == "__main__":
    main()
