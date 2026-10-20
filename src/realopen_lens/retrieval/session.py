"""Explicit search lifetime; indexes and source validity are checked on every call."""

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from realopen_lens.learning.feedback import RecordedSearch

from realopen_lens.retrieval.search import SearchResult, search_collection
from realopen_lens.runtime.resident import ResidentQueryEncoder


class SearchSession:
    """Reuse lazy query encoders without caching collection contents or results."""

    def __init__(
        self,
        *,
        collection: str = "personal",
        index_dir: Path | None = None,
        models: Path = Path("models"),
        threads: int = 2,
        timeout: float = 120,
    ):
        if not 1 <= threads <= 8:
            raise ValueError("Use 1–8 inference threads")
        self.collection, self.index_dir = collection, index_dir
        self.models, self.threads = models, threads
        self._encoder = ResidentQueryEncoder(timeout=timeout)
        self._closed = False

    @property
    def worker_pid(self) -> int | None:
        return self._encoder.pid

    def search(self, query: str, **options: Any) -> list[SearchResult]:
        if self._closed:
            raise RuntimeError("Search session is closed")
        return search_collection(
            query,
            collection=self.collection,
            index_dir=self.index_dir,
            models=self.models,
            threads=self.threads,
            query_encoder=self._encoder,
            **options,
        )

    def search_recorded(self, query: str, **options: Any) -> "RecordedSearch":
        """Opt into saving a search for explicit relevance feedback."""
        from realopen_lens.learning.feedback import search_and_record

        if self._closed:
            raise RuntimeError("Search session is closed")
        return search_and_record(
            query,
            collection=self.collection,
            index_dir=self.index_dir,
            models=self.models,
            threads=self.threads,
            query_encoder=self._encoder,
            **options,
        )

    def close(self):
        self._closed = True
        self._encoder.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
