"""CPU ONNX adapters with explicit local preprocessing and bounded image decoding."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

ort.disable_telemetry_events()


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Encoder returned non-finite embeddings")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Encoder returned a zero or invalid embedding norm")
    return vectors / norms


def image_rgb(path: Path, max_pixels: int = 24_000_000) -> Image.Image:
    if max_pixels < 1:
        raise ValueError("Decoded pixel limit must be positive")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"Image exceeds 64 MiB encoded size: {path.name}")
    with Image.open(path) as source:
        if source.format not in {"JPEG", "MPO", "PNG", "WEBP", "TIFF", "BMP"}:
            raise ValueError(f"Unsupported image content format: {source.format}")
        if source.width * source.height > max_pixels:
            raise ValueError(f"Image exceeds {max_pixels} decoded pixels: {path.name}")
        oriented = ImageOps.exif_transpose(source)
        try:
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                try:
                    background = Image.new("RGBA", rgba.size, "white")
                    background.alpha_composite(rgba)
                    result = background.convert("RGB")
                    background.close()
                finally:
                    rgba.close()
            else:
                result = oriented.convert("RGB")
            # Metadata can contain a second orientation, stale DPI, or large ICC/XMP
            # payloads. Intermediate OCR images need only the oriented pixels.
            result.info.clear()
            return result
        finally:
            oriented.close()


class Encoder:
    def __init__(self, graph: Path, threads: int):
        if not 1 <= threads <= 32:
            raise ValueError("Inference threads must be between 1 and 32")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(graph), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        names = [output.name for output in self.session.get_outputs()]
        required = {item.name for item in self.session.get_inputs()}
        return dict(
            zip(
                names,
                self.session.run(None, {k: v for k, v in inputs.items() if k in required}),
                strict=True,
            )
        )


class TextEncoder(Encoder):
    def __init__(self, graph: Path, tokenizer: Path, kind: str, threads: int):
        super().__init__(graph, threads)
        self.kind = kind
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_truncation(max_length=64 if kind == "siglip2" else 512)
        self.tokenizer.enable_padding(
            length=64 if kind == "siglip2" else None,
            pad_id=0 if kind == "siglip2" else 1,
            pad_token="<pad>",
        )

    def encode(self, texts: list[str], *, passage: bool = False) -> np.ndarray:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Text encoding requires nonempty strings")
        if self.kind == "siglip2":
            texts = [text.lower() for text in texts]
        else:
            texts = [("passage: " if passage else "query: ") + text for text in texts]
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.array([item.ids for item in encoded], dtype=np.int64)
        mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
        outputs = self.run(
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        )
        if self.kind == "siglip2":
            vectors = outputs["pooler_output"]
        else:
            hidden = outputs["last_hidden_state"]
            vectors = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        return normalize(vectors)

    def passages(self, text: str) -> list[str]:
        """Split using the actual tokenizer; reserve room for prefix and special tokens."""
        # Tokenizer truncation is disabled temporarily to avoid losing long OCR content.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            return [self.tokenizer.decode(ids[i : i + 440]) for i in range(0, len(ids), 400)]
        finally:
            self.tokenizer.enable_truncation(max_length=512)
            self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")


class VisionEncoder(Encoder):
    def encode(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            raise ValueError("Vision encoding requires at least one image")
        pixels = []
        for path in paths:
            image = image_rgb(path)
            try:
                resized = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
                array = np.asarray(resized, dtype=np.float32) * np.float32(1 / 255)
                pixels.append(((array - 0.5) / 0.5).transpose(2, 0, 1))
            finally:
                image.close()
        return normalize(self.run({"pixel_values": np.stack(pixels)})["pooler_output"])
