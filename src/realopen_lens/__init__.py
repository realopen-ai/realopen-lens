"""Private, offline image indexing and retrieval engine."""

__version__ = "0.1.0"

# Registration reads no photos and performs no inference. Decode only the primary
# image using one native thread, respecting the existing per-image worker bounds.
from pi_heif import register_heif_opener

register_heif_opener(thumbnails=False, depth_images=False, aux_images=False, decode_threads=1)
