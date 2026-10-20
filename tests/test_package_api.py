"""Documented Python entry points and bundled resources survive package moves."""


def test_public_collection_entry_points_remain_available():
    from realopen_lens.indexing import IndexResult, collection_status, index_collection
    from realopen_lens.indexing import service
    from realopen_lens.search import CandidateDepths, SearchResult, SearchSession, search_collection
    from realopen_lens.retrieval.session import SearchSession as Session
    from realopen_lens.retrieval import search

    assert (IndexResult, collection_status, index_collection) == (
        service.IndexResult,
        service.collection_status,
        service.index_collection,
    )
    assert (SearchResult, search_collection) == (search.SearchResult, search.search_collection)
    assert CandidateDepths().as_dict() == {"visual": 50, "semantic": 50, "lexical": 50}
    assert SearchSession is Session


def test_bundled_model_manifests_load_from_the_models_package():
    from realopen_lens.models import registry
    from realopen_lens.ocr.regions import detector_spec

    assert {"siglip2", "e5"}.issubset(registry()["models"])
    detector = detector_spec()
    assert detector["size"] > 0 and len(detector["sha256"]) == 64
