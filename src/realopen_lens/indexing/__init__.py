"""Persistent collection indexing operations."""

from .service import IndexResult as IndexResult
from .service import index_collection as index_collection
from .service import collection_status as collection_status

__all__ = ["IndexResult", "index_collection", "collection_status"]
