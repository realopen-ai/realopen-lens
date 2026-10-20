"""Public collection-search entry points; implementation lives in retrieval."""

from realopen_lens.retrieval.search import SearchResult as SearchResult
from realopen_lens.retrieval.search import search_collection as search_collection
from realopen_lens.retrieval.candidates import CandidateDepths as CandidateDepths

__all__ = ["CandidateDepths", "SearchResult", "search_collection"]
