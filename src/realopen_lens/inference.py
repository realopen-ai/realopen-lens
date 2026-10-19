"""CPU ONNX adapters with explicit local preprocessing and bounded image decoding."""

from pathlib import Path


import onnxruntime as ort


from PIL import Image, ImageOps

ort.disable_telemetry_events()


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
