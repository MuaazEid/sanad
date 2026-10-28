import pytest

from sanad.retrieve import HybridRetriever


def test_hybrid_retrieves_bilingual_content(retriever):
    ar = retriever.search("متى تأسست جامعة الجزيرة؟", top_k=3)
    en = retriever.search("How many stations does the Riyadh Metro have?", top_k=3)
    assert any(h.chunk.doc_id == "ar-gezira" for h in ar)
    assert any(h.chunk.doc_id == "en-metro" for h in en)


def test_lexical_mode_returns_bm25_only(retriever):
    hits = retriever.search("Riyadh Metro", top_k=3, mode="lexical")
    assert hits
    assert all(h.bm25_rank is not None and h.dense_rank is None for h in hits)


def test_hybrid_records_both_signal_ranks(retriever):
    hits = retriever.search("Riyadh Metro 85 stations", top_k=3)
    assert hits[0].bm25_rank is not None
    assert hits[0].dense_rank is not None


def test_top_k_is_respected_and_ids_unique(retriever):
    hits = retriever.search("Riyadh", top_k=2)
    assert len(hits) <= 2
    assert len({h.chunk.id for h in hits}) == len(hits)


def test_invalid_mode_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_hybrid_degrades_to_lexical_without_vectors(store, embedder):
    store.conn.execute("DELETE FROM embeddings")
    store.conn.commit()
    hits = HybridRetriever(store, embedder).search("Riyadh Metro", top_k=3)
    assert hits
    assert all(h.dense_rank is None for h in hits)


def test_rrf_rewards_items_found_by_both_signals(retriever):
    hits = retriever.search("Riyadh Metro 85 stations", top_k=5)
    top = hits[0]
    assert top.bm25_rank is not None and top.dense_rank is not None


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("", top_k=5) == []
    assert retriever.search("؟،؛", top_k=5) == []
