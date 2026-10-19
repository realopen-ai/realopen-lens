"""Public collection-search entry points; implementation lives in retrieval."""

from realopen_lens.retrieval.search import SearchResult as SearchResult
from realopen_lens.retrieval.search import search_collection as search_collection

__all__ = ["SearchResult", "search_collection"]
