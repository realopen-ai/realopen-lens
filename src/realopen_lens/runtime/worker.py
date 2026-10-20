"""Internal managed indexing/query workers. Invoke public operations through the CLI."""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import signal
import sys


def interrupted(signum, frame):
    raise KeyboardInterrupt


def configure_logging(config):
    if config.get("log", False):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )


def main():
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--stage", choices=["metadata", "ocr", "vision", "semantic"])
    parser.add_argument("--query", action="store_true")
    args = parser.parse_args()
    from realopen_lens.runtime.offline import enforce

    enforce()
    if args.query:
        from realopen_lens.encoders.onnx import TextEncoder
        from realopen_lens.models.manager import artifact_path

        request = json.load(sys.stdin)
        root, kind = Path(request["models"]), request["kind"]
        precision = request.get("siglip_precision", "int8") if kind == "siglip2" else "int8"
        if precision not in {"int8", "fp32"}:
            raise ValueError("SigLIP2 precision must be int8 or fp32")
        model = TextEncoder(
            artifact_path(root, kind, f"text_{precision}"),
            artifact_path(root, kind, "tokenizer"),
            kind,
            request["threads"],
        )
        print(json.dumps(model.encode([request["query"]])[0].tolist()))
    elif args.request:
        from realopen_lens.indexing.service import run_index

        try:
            request = json.loads(args.request.read_text())
            configure_logging(request)
            result = run_index(request)
            print(json.dumps(asdict(result)))
        except (Exception, KeyboardInterrupt) as error:
            print(f"Indexing failed: {type(error).__name__}: {error}", file=sys.stderr)
            raise SystemExit(1) from error
    else:
        from realopen_lens.storage.catalog import Catalog
        from realopen_lens.indexing.processing import process_stage

        catalog = Catalog(args.store)
        try:
            configure_logging(json.loads(catalog.job(args.job)["config"]))
            process_stage(catalog, args.job, args.stage)
        finally:
            catalog.close()


if __name__ == "__main__":
    main()
