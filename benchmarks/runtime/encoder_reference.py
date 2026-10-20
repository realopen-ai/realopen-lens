"""Optional PyTorch reference adapters; never imported by production code."""

from pathlib import Path
from typing import Any


class ReferenceEncoder:
    def __init__(self, directory: Path, kind: str, threads: int, siglip_lowercase: bool = False):
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.kind = kind
        self.siglip_lowercase = siglip_lowercase
        self.model = AutoModel.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        if kind == "vision":
            self.processor = AutoImageProcessor.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=False,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
        self.last_inputs: dict[str, Any] = {}

    def encode(self, inputs: list, *, passage: bool = False):
        import numpy as np
        from PIL import Image

        torch = self.torch
        if self.kind == "vision":
            with Image.open(inputs[0]) as image:
                values = self.processor(images=image, return_tensors="pt")
            self.last_inputs = {"pixel_values": values["pixel_values"].numpy()}
            with torch.inference_mode():
                vectors = self.model.get_image_features(**values)
        else:
            texts = inputs
            if self.kind == "siglip2" and self.siglip_lowercase:
                texts = [text.lower() for text in texts]
            if self.kind == "e5":
                texts = [("passage: " if passage else "query: ") + text for text in texts]
            # Record official tokenizer IDs so preprocessing differences remain visible.
            values = self.tokenizer(
                texts,
                max_length=512 if self.kind == "e5" else 64,
                truncation=True,
                padding=True if self.kind == "e5" else "max_length",
                return_tensors="pt",
            )
            self.last_inputs = {key: value.numpy() for key, value in values.items()}
            with torch.inference_mode():
                if self.kind == "siglip2":
                    vectors = self.model.get_text_features(**values)
                else:
                    hidden = self.model(**values).last_hidden_state
                    mask = values["attention_mask"].unsqueeze(-1)
                    vectors = (hidden * mask).sum(1) / mask.sum(1)
        result = vectors.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(result, axis=-1, keepdims=True)
        if not np.isfinite(result).all() or not (norms > 0).all():
            raise ValueError("Invalid reference embedding")
        return result / norms
