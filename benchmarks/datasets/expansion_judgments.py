"""Freeze reviewed expansion intents while retaining legacy folds as diagnostics."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from benchmarks.datasets.downloads import digest
from benchmarks.datasets.photos import save_json
from benchmarks.datasets.relevance import load
from benchmarks.datasets.training_review import apply


def assemble(corpus: dict, legacy: dict, review: dict) -> dict:
    images = {row["path"]: row for row in corpus["images"]}
    observations = review["observations"]
    if not observations or any(
        path not in images
        or images[path]["sha256"] != observation["sha256"]
        or not observation["description"].strip()
        for path, observation in observations.items()
    ):
        raise ValueError("Review must identify the exact inspected images and observations")
    provenance = review["review"]
    if provenance.get("status") != "reviewed":
        raise ValueError("Expansion requires reviewed relevance decisions")
    queries = deepcopy(legacy["queries"])
    for query in queries:
        query["study_role"] = "train" if query["split"] == "train" else "diagnostic"
    families = {q["family"] for q in queries}
    for intent in review["intents"]:
        family = intent["id"]
        if family in families or set(intent["translations"]) != {"en", "fr", "ar"}:
            raise ValueError("Expansion needs unique families with all three translations")
        families.add(family)
        targets = intent["positives"]
        if not targets or any(path not in observations for path in targets):
            raise ValueError("Every intent needs visually inspected positive images")
        split = images[targets[0]]["split"]
        kind = images[targets[0]]["category"]
        if any(images[p]["split"] != split or images[p]["category"] != kind for p in targets):
            raise ValueError("Positive images must share source fold and kind")
        groups = {images[path]["duplicate_group"] for path in targets}
        labels = []
        for path, observation in observations.items():
            image = images[path]
            if image["split"] != split:
                continue
            if path in targets:
                grade, reason = 2, "Direct inspected match: " + observation["description"]
            elif image["duplicate_group"] in groups or path in intent.get("unjudged", []):
                # Same-work variants can contain partial evidence; absence is not a negative.
                continue
            else:
                grade, reason = 0, (
                    "Distinct inspected required subject; this image instead shows: "
                    + observation["description"]
                )
            labels.append({"image_sha256": image["sha256"], "grade": grade, "reason": reason})
        for language, text in intent["translations"].items():
            queries.append(
                {
                    "id": family + "-" + language,
                    "family": family,
                    "text": text,
                    "language": language,
                    "kind": kind,
                    "split": split,
                    "study_role": split,
                    "source_exposure": "fresh-v2",
                    "review": provenance,
                    "judgments": deepcopy(labels),
                }
            )
    return {"version": 1, "queries": queries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--legacy-corpus", type=Path, required=True)
    parser.add_argument("--legacy-judgments", type=Path, required=True)
    parser.add_argument("--training-review", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if corpus.get("protocol_sha256") != digest(args.protocol):
        raise ValueError("Corpus identifies a different study protocol")
    old_corpus, legacy = load(args.legacy_corpus, args.legacy_judgments)
    legacy = apply(legacy, old_corpus, json.loads(args.training_review.read_text(encoding="utf-8")))
    bundle = assemble(corpus, legacy, json.loads(args.review.read_text(encoding="utf-8")))
    bundle.update(
        corpus_sha256=digest(args.corpus),
        protocol_sha256=digest(args.protocol),
        review_sha256=digest(args.review),
        legacy_judgments_sha256=digest(args.legacy_judgments),
        training_review_sha256=digest(args.training_review),
    )
    if args.output.exists():
        raise ValueError("Refusing to overwrite frozen judgments")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".tmp")
    try:
        save_json(pending, bundle)
        load(args.corpus, pending)
        pending.replace(args.output)
    finally:
        pending.unlink(missing_ok=True)
    print(json.dumps({"queries": len(bundle["queries"]), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
