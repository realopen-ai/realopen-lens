"""Private newline-framed query worker; stdin EOF ends the owned process."""

import json
from pathlib import Path
import sys

from realopen_lens.runtime.offline import enforce


def serve():
    enforce()
    from realopen_lens.encoders.onnx import TextEncoder
    from realopen_lens.models.manager import artifact_path

    identity = None
    encoders = {}
    while line := sys.stdin.readline(65537):
        try:
            if len(line) > 65536 or not line.endswith("\n"):
                raise ValueError("Query request exceeded its protocol limit")
            request = json.loads(line)
            root = Path(request["models"])
            kind, threads, precision = (
                request["kind"],
                request["threads"],
                request["siglip_precision"],
            )
            text = request["query"]
            if (
                kind not in {"siglip2", "e5"}
                or precision not in {"int8", "fp32"}
                or not 1 <= threads <= 8
                or not isinstance(text, str)
                or not text.strip()
                or len(text) > 4096
            ):
                raise ValueError("Invalid query request")
            current = (str(root), threads, precision)
            if identity is not None and current != identity:
                raise ValueError("Worker configuration changed")
            identity = current
            if kind not in encoders:
                selected = precision if kind == "siglip2" else "int8"
                encoders[kind] = TextEncoder(
                    artifact_path(root, kind, f"text_{selected}"),
                    artifact_path(root, kind, "tokenizer"),
                    kind,
                    threads,
                )
            vector = encoders[kind].encode([text])[0].tolist()
            print(json.dumps({"vector": vector}), flush=True)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {str(error)[:1500]}"}), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
