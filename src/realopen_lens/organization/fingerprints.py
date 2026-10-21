"""Versioned still-image fingerprints; no model inference or file identity changes."""

import hashlib
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

POLICY = "oriented-rgba-v1/dhash256-rgb16-v1"


def fingerprint(path: Path) -> dict:
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are not eligible for duplicate grouping")
        if source.width * source.height > 40_000_000:
            raise ValueError("Duplicate fingerprint exceeds 40 megapixel budget")
        with ImageOps.exif_transpose(source) as oriented:
            rgba = np.array(oriented.convert("RGBA"), dtype=np.uint8)
    # Hidden RGB under fully transparent pixels does not change visible content.
    rgba[rgba[:, :, 3] == 0, :3] = 0
    height, width = rgba.shape[:2]
    digest = hashlib.sha256(b"lens-pixels-v1\0" + struct.pack(">II", width, height))
    for row in rgba:
        digest.update(row.tobytes())
    with Image.fromarray(rgba) as normalized:
        with Image.new("RGBA", normalized.size, "white") as background:
            background.alpha_composite(normalized)
            rgb = background.convert("RGB")
        try:
            tiny = np.asarray(rgb.resize((16, 16), Image.Resampling.LANCZOS))
            gray = np.asarray(rgb.convert("L").resize((17, 16), Image.Resampling.LANCZOS))
        finally:
            rgb.close()
    bits = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex()
    return {
        "pixels": digest.hexdigest(),
        "width": width,
        "height": height,
        "dhash": bits,
        "rgb16": tiny.tobytes().hex(),
        "contrast": float(tiny.astype(np.float32).std(axis=(0, 1)).mean()),
    }


def near_distance(left: dict, right: dict) -> dict | None:
    """Conservative whole-frame suggestions, never proof of equivalent content."""
    if left["pixels"] == right["pixels"]:
        return None
    aspect = abs((left["width"] / left["height"]) / (right["width"] / right["height"]) - 1)
    if aspect > 0.02 or min(left["contrast"], right["contrast"]) < 8:
        return None
    hamming = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
    if hamming > 8:
        return None
    a = np.frombuffer(bytes.fromhex(left["rgb16"]), dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(bytes.fromhex(right["rgb16"]), dtype=np.uint8).astype(np.int16)
    mae = float(np.abs(a - b).mean())
    if mae > 12:
        return None
    return {"hamming": hamming, "rgb_mae": mae, "aspect_delta": aspect}
