"""Pinned model artifacts, explicit setup, and local integrity checks."""

from .manager import registry as registry
from .manager import digest as digest
from .manager import artifacts as artifacts
from .manager import artifact_path as artifact_path
from .manager import verify as verify
from .manager import fetch as fetch
from .manager import fetch_text_detector as fetch_text_detector

__all__ = [
    "registry",
    "digest",
    "artifacts",
    "artifact_path",
    "verify",
    "fetch",
    "fetch_text_detector",
]
