"""Native synthetic workflow check; labels validate plumbing, never retrieval quality."""

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from realopen_lens.indexing.service import index_collection
from realopen_lens.learning import dataset, feedback, registry, workflow
from realopen_lens.models.manager import digest
from realopen_lens.storage.artifacts import atomic_bytes


def run(output: Path, models: Path, precision: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    documents = [
        ("invoice", "Invoice INV-004231\nCustomer: Example Studio\nTotal: 125 euros"),
        ("receipt", "Receipt R-9821\nCoffee beans: 12 euros\nThank you for your order"),
        ("terminal", "Deployment log\nERROR ERR_42: PostgreSQL connection refused\nRetry pending"),
        ("diagram", "Architecture diagram\nBrowser -> API -> local database\nOffline storage"),
        ("meeting", "Meeting notes\nProject review: Thursday\nAction: update documentation"),
        ("form", "Registration form 2048\nName: Example User\nDepartment: Research"),
    ]
    for index, (name, text) in enumerate(documents):
        image = Image.new("RGB", (900, 500), (250, 250 - index, 245))
        ImageDraw.Draw(image).multiline_text(
            (40, 60), text, font=ImageFont.load_default(size=32), fill="black", spacing=20
        )
        image.save(images / f"{name}.png")
    store = output / "index"
    phases: list[str] = []

    def progress(value):
        phase = value["phase"]
        if not phases or phases[-1] != phase:
            phases.append(phase)
            print(json.dumps({"phase": phase}), flush=True)

    indexed = index_collection(
        images,
        index_dir=store,
        collection="smoke",
        models=models,
        languages="eng",
        siglip_precision=precision,
        on_progress=progress,
    )
    review: dict[str, Any] = {
        "version": 1,
        "reviewer": "synthetic infrastructure check; not human judgments",
        "independent_holdout": False,
        "runs": {},
        "image_groups": {},
    }
    for split, query, positive, negative in (
        ("train", "invoice INV-004231", "invoice", "receipt"),
        ("validation", "PostgreSQL ERR_42", "terminal", "diagram"),
        ("test", "meeting notes Thursday", "meeting", "form"),
    ):
        saved = feedback.search_and_record(
            query, index_dir=store, models=models, collection="smoke", limit=50
        )
        for result in saved.results:
            name = Path(result.paths[0]).stem
            if name in (positive, negative):
                feedback.judge(
                    saved.run_id,
                    result.image_id,
                    "relevant" if name == positive else "irrelevant",
                    index_dir=store,
                )
                review["image_groups"][result.image_id] = name
        review["runs"][saved.run_id] = {"family": query, "split": split}
    source = output / "feedback.jsonl"
    count = feedback.export_feedback(source, index_dir=store)
    if count != 6:
        raise ValueError("Synthetic searches must expose all six intended labeled pairs")
    review.update(feedback_file=source.name, feedback_sha256=digest(source))
    review_path = output / "review.json"
    atomic_bytes(review_path, dataset.canonical(review))
    prepared = dataset.prepare(store, review_path, output / "dataset", progress)
    trained = workflow.train(output / "dataset/dataset.json", store, models, progress)
    if trained["status"] != "completed":
        raise ValueError("Native production evaluation did not complete")
    registry.change(store, "smoke", trained["id"], acknowledge_experimental=True)
    active = registry.active(store, "smoke")
    registry.change(store, "smoke", None)
    registry.change(store, "smoke", trained["id"], acknowledge_experimental=True, rollback=True)
    registry.change(store, "smoke", None)
    summary = {
        "purpose": "synthetic infrastructure validation; no real quality claims",
        "precision": precision,
        "images": indexed.images,
        "dataset": prepared,
        "trained": trained,
        "activated": str(active),
        "restored_hybrid": True,
        "phases": phases,
    }
    atomic_bytes(output / "summary.json", dataset.canonical(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    args = parser.parse_args()
    run(args.output, args.models, args.precision)
