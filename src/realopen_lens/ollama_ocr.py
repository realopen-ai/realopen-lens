"""Explicit local GLM-OCR backend using Ollama's native API and bounded generation."""

import base64
import http.client
import io
import json
import math
import re
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw

from realopen_lens.inference import image_rgb


class OllamaOCR:
    """One local model, no proxies/redirects/cloud routes or inference-time downloads."""

    def __init__(
        self,
        model: str = "glm-ocr:q8_0",
        port: int = 11434,
        threads: int = 2,
        timeout: float = 90,
        max_tokens: int = 4096,
        context: int = 8192,
        expected_digest: str | None = None,
    ):
        if not 1 <= port <= 65535 or not 1 <= threads <= 32:
            raise ValueError("Invalid Ollama port or thread count")
        if not math.isfinite(timeout) or timeout <= 0 or not 1 <= max_tokens <= 8192:
            raise ValueError("Invalid OCR timeout or output token budget")
        if not 1024 <= context <= 16384 or max_tokens >= context:
            raise ValueError("Output token budget must fit within the context budget")
        if not model or model.endswith(":cloud"):
            raise ValueError("An explicitly installed local GLM-OCR model is required")
        self.model, self.port, self.timeout = model, port, timeout
        self.options = {
            "temperature": 0,
            "seed": 20261019,
            "num_thread": threads,
            "num_ctx": context,
            "num_predict": max_tokens,
        }
        self.info = self.inspect()
        if expected_digest and self.info["digest"] != expected_digest:
            raise ValueError("Ollama model digest changed since run creation")

    def request(self, path: str, payload: dict | None = None) -> dict:
        # HTTPConnection never consults proxy environment variables or follows redirects.
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            body = json.dumps(payload).encode() if payload is not None else None
            connection.request(
                "POST" if payload is not None else "GET",
                path,
                body=body,
                headers={"Content-Type": "application/json", "Connection": "close"},
            )
            response = connection.getresponse()
            content = response.read(8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024:
                raise ValueError("Ollama response exceeded 8 MiB")
            if response.status != 200:
                raise ValueError(
                    f"Local Ollama HTTP {response.status}: "
                    f"{content[:1000].decode(errors='replace')}"
                )
            result = json.loads(content)
            if not isinstance(result, dict) or result.get("error"):
                raise ValueError(f"Invalid Ollama response: {str(result)[:1000]}")
            return result
        except (OSError, http.client.HTTPException) as error:
            raise ConnectionError(
                f"Local Ollama request failed at 127.0.0.1:{self.port}: {error}"
            ) from error
        finally:
            connection.close()

    def inspect(self) -> dict:
        version = self.request("/api/version")["version"]
        tags = self.request("/api/tags").get("models", [])
        found = next(
            (r for r in tags if r.get("name") == self.model or r.get("model") == self.model), None
        )
        if not found:
            raise ValueError(
                f"Local model {self.model!r} is missing; explicitly run 'ollama pull {self.model}'"
            )
        shown = self.request("/api/show", {"model": self.model})
        if shown.get("remote_model") or shown.get("remote_host") or found.get("remote_model"):
            raise ValueError("Remote/cloud models are not supported by local OCR")
        digest = found.get("digest", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Ollama returned an invalid local model digest")
        if "vision" not in shown.get("capabilities", []):
            raise ValueError("The selected Ollama model has no vision capability")
        return {
            "server_version": version,
            "model": self.model,
            "digest": digest,
            "bytes_on_disk": found.get("size"),
            "details": shown.get("details", {}),
            "capabilities": shown.get("capabilities", []),
            "parameters": shown.get("parameters"),
            "options": self.options,
            "endpoint": f"http://127.0.0.1:{self.port}/api/generate",
        }

    def loaded(self) -> dict:
        rows = self.request("/api/ps").get("models", [])
        for row in rows:
            if row.get("name") == self.model or row.get("model") == self.model:
                if row.get("digest") != self.info["digest"]:
                    raise ValueError("Loaded Ollama model digest changed")
                return {
                    k: row.get(k) for k in ("name", "digest", "size", "size_vram", "context_length")
                }
        return {}

    def generate(self, image: Image.Image, *, warmup: bool = False) -> dict:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        result = self.request(
            "/api/generate",
            {
                "model": self.model,
                "prompt": "Text Recognition:",
                "images": [encoded],
                "stream": False,
                "keep_alive": "5m",
                "options": {**self.options, **({"num_predict": 32} if warmup else {})},
            },
        )
        return parse_response(result)

    def warmup(self) -> dict:
        with Image.new("RGB", (320, 96), "white") as image:
            ImageDraw.Draw(image).text((20, 35), "Warmup", fill="black", font_size=24)
            return self.generate(image, warmup=True)

    def extract(self, path: Path) -> dict:
        with image_rgb(path) as image:
            original_size = image.size
            # Check the original pixels: resizing can erase tiny text. A truly
            # uniform RGB image has no visible glyphs and needs no generation.
            extrema = cast(tuple[tuple[float, float], ...], image.getextrema())
            uniform = all(low == high for low, high in extrema)
            image.thumbnail((2400, 2400))
            result: dict[str, Any]
            if uniform:
                result = {
                    "text": "",
                    "confidence": None,
                    "words": None,
                    "warnings": [],
                    "done_reason": None,
                    "truncated": False,
                    "inference_skipped": "uniform_pixels",
                    "ollama_metrics": {},
                    "geometry": "No visible text in exactly uniform RGB pixels",
                }
            else:
                result = self.generate(image)
            result["ocr_size"] = list(image.size)
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) > 1:
                result["warnings"].append("Only first frame/page processed")
        result.update(original_size=list(original_size), loaded_model=self.loaded())
        return result

    def close(self) -> None:
        self.request("/api/generate", {"model": self.model, "keep_alive": 0})


def parse_response(result: dict[str, Any]) -> dict:
    if result.get("done") is not True or not isinstance(result.get("response"), str):
        raise ValueError("Incomplete or malformed Ollama generation")
    text = result["response"]
    reason = result.get("done_reason")
    if reason not in {"stop", "length"}:
        raise ValueError(f"Unexpected Ollama generation stop reason: {reason}")
    warnings = []
    if reason == "length":
        warnings.append("Generation exhausted token/context budget; output may be truncated")
    if not text.strip():
        warnings.append("No recognized text; absence of text is not established")
    return {
        "text": text,
        "confidence": None,
        "words": None,
        "warnings": warnings,
        "done_reason": reason,
        "truncated": reason == "length",
        "ollama_metrics": {
            key: result.get(key)
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
        },
        "ollama_metrics_units": "durations in nanoseconds; counts in tokens",
        "geometry": "No word boxes supplied by this image-to-text request",
    }
