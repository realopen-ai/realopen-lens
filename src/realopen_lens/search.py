"""Public collection-search entry points; implementation lives in retrieval."""

from realopen_lens.retrieval.search import SearchResult as SearchResult
from realopen_lens.retrieval.search import search_collection as search_collection
from realopen_lens.retrieval.candidates import CandidateDepths as CandidateDepths
from realopen_lens.retrieval.session import SearchSession as SearchSession

__all__ = ["CandidateDepths", "SearchResult", "SearchSession", "search_collection"]
