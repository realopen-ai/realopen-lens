"""One subprocess per model stage, keeping heavyweight runtimes out of the parent."""

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from realopen_lens.runtime.offline import enforce


def ocr(config: dict[str, Any]) -> dict[str, Any]:
    from realopen_lens.ocr.tesseract import extract

    records, times = [], []
    environment = {}
    for image_id in config["images"]:
        start = perf_counter()
        result = extract(Path(config["data"]) / image_id, config["languages"], config["threads"])
        environment = {key: result[key] for key in ("version", "language_pack_sha256")}
        records.append({"image_id": image_id, **result})
        times.append(perf_counter() - start)
    return {"records": records, "image_seconds": times, **environment}


def embeddings(stage: str, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from realopen_lens.encoders.onnx import TextEncoder, VisionEncoder
    from realopen_lens.models.manager import artifact_path

    output = Path(config["output"])
    root = Path(config["models"])
    precision = config["precision"]
    start = perf_counter()
    if stage == "vision":
        encoder = VisionEncoder(
            artifact_path(root, "siglip2", f"vision_{precision}"), config["threads"]
        )
        load_seconds = perf_counter() - start
        paths = [Path(config["data"]) / image_id for image_id in config["images"]]
        batch = config["batch_size"]
        times, vectors = [], []
        for i in range(0, len(paths), batch):
            start = perf_counter()
            vectors.append(encoder.encode(paths[i : i + batch]))
            times.append(perf_counter() - start)
        np.save(output / "vision.npy", np.concatenate(vectors))
        return {
            "load_seconds": load_seconds,
            "batch_seconds": times,
            "images_per_second": len(paths) / sum(times),
        }

    kind = "siglip2" if stage == "visual_queries" else "e5"
    text_encoder = TextEncoder(
        artifact_path(root, kind, f"text_{precision}"),
        artifact_path(root, kind, "tokenizer"),
        kind,
        config["threads"],
    )
    load_seconds = perf_counter() - start
    texts = [query["text"] for query in config["queries"]]
    # Time first and repeated execution of each query, excluding model load and retrieval.
    vectors, cold_times, warm_times = [], [], []
    for text in texts:
        start = perf_counter()
        vector = text_encoder.encode([text])
        cold_times.append(perf_counter() - start)
        vectors.append(vector)
        timings = []
        for _ in range(config["repeats"]):
            start = perf_counter()
            text_encoder.encode([text])
            timings.append(perf_counter() - start)
        warm_times.append(timings)
    if vectors:
        np.save(output / f"{stage}.npy", np.concatenate(vectors))

    result: dict[str, Any] = {
        "load_seconds": load_seconds,
        "query_first_seconds": cold_times,
        "query_repeat_seconds": warm_times,
    }
    if kind == "e5":
        records = json.loads((output / "ocr.json").read_text())["records"]
        passages = [
            {"image_id": record["image_id"], "text": chunk}
            for record in records
            if record["text"]
            for chunk in text_encoder.passages(record["text"])
        ]
        start = perf_counter()
        passage_vectors = [
            text_encoder.encode([passage["text"]], passage=True) for passage in passages
        ]
        result["passage_seconds"] = perf_counter() - start
        result["passage_count"] = len(passages)
        if passage_vectors:
            np.save(output / "passages.npy", np.concatenate(passage_vectors))
        (output / "passages.json").write_text(
            json.dumps(passages, ensure_ascii=False), encoding="utf-8"
        )
    return result


def main() -> None:
    enforce()
    stage, config_path = sys.argv[1:]
    config = json.loads(Path(config_path).read_text())
    result = ocr(config) if stage == "ocr" else embeddings(stage, config)
    (Path(config["output"]) / f"{stage}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
