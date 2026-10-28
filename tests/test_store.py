import pytest

from sanad.embed import HashingEmbedder
from sanad.store import EmbeddingModelMismatch, Store, StoreError
from sanad.text import chunk_text


def _ingest(store, embedder, doc_id="ar-x", text="جامعة الجزيرة تقع في ود مدني.",
            lang="ar", title="جامعة الجزيرة", **kw):
    return store.ingest(doc_id=doc_id, title=title, lang=lang, text=text,
                        chunks=chunk_text(text), embedder=embedder, **kw)


def test_ingest_creates_documents_chunks_vectors(tmp_path):
    s = Store(tmp_path / "x.db")
    e = HashingEmbedder(dim=64)
    stats = _ingest(s, e)
    assert stats.chunks >= 1 and not stats.skipped
    counts = s.stats()
    assert counts["documents"] == 1
    assert counts["chunks"] == counts["vectors"] == stats.chunks
    assert counts["embedding_model"] == e.name
    s.close()


def test_reingest_unchanged_skips(tmp_path):
    s = Store(tmp_path / "x.db")
    e = HashingEmbedder(dim=64)
    _ingest(s, e)
    assert _ingest(s, e).skipped
    assert _ingest(s, e, force=True).skipped is False
    s.close()


def test_reingest_replaces_old_chunks_and_fts(tmp_path):
    s = Store(tmp_path / "x.db")
    e = HashingEmbedder(dim=64)
    _ingest(s, e, text="The old text mentions zarqaa uniqueword.")
    _ingest(s, e, text="The new text mentions nothing similar.")
    assert s.search_bm25("uniqueword") == []
    assert s.search_bm25("similar") != []
    assert s.stats()["documents"] == 1
    s.close()


def test_bm25_is_normalization_insensitive(tmp_path):
    s = Store(tmp_path / "x.db")
    _ingest(s, HashingEmbedder(dim=64), text="تأسست جامعة الجزيرة في 9 نوفمبر 1975.")
    for query in ("الجزيرة", "الجزيره", "جَامِعَة"):
        assert s.search_bm25(query), query
    s.close()


def test_bm25_language_filter(tmp_path):
    s = Store(tmp_path / "x.db")
    e = HashingEmbedder(dim=64)
    _ingest(s, e, doc_id="ar-1", lang="ar", text="مترو الرياض فيه 85 محطة.")
    _ingest(s, e, doc_id="en-1", lang="en", title="Riyadh Metro",
            text="The Riyadh Metro has 85 stations.")
    ar = s.search_bm25("Riyadh", lang="ar")
    en = s.search_bm25("Riyadh", lang="en")
    assert ar == []
    assert en and en[0].doc_id == "en-1"
    s.close()


def test_bm25_survives_punctuation_and_quotes(tmp_path):
    s = Store(tmp_path / "x.db")
    _ingest(s, HashingEmbedder(dim=64))
    assert s.search_bm25('"مترو" OR *') == []
    assert s.search_bm25("؟،؛") == []


def test_dense_search_orders_by_similarity(tmp_path):
    s = Store(tmp_path / "x.db")
    e = HashingEmbedder(dim=512)
    _ingest(s, e, doc_id="metro", text="The Riyadh Metro has six lines and 85 stations.")
    _ingest(s, e, doc_id="water", text="Desalination removes salt from sea water.")
    q = e.embed(["Riyadh Metro stations"])[0]
    hits = s.search_dense(q, e.name, limit=5)
    assert hits[0].doc_id == "metro"
    s.close()


def test_embedding_model_mismatch_on_ingest(tmp_path):
    s = Store(tmp_path / "x.db")
    _ingest(s, HashingEmbedder(dim=64))
    with pytest.raises(EmbeddingModelMismatch):
        _ingest(s, HashingEmbedder(dim=128), force=True)
    s.close()


def test_embedding_model_mismatch_on_query(tmp_path):
    s = Store(tmp_path / "x.db")
    _ingest(s, HashingEmbedder(dim=64))
    with pytest.raises(EmbeddingModelMismatch):
        s.search_dense([0.1] * 64, "ollama:bge-m3")
    s.close()


def test_embedder_vector_count_must_match(tmp_path):
    class Bad:
        name = "bad:1"
        def embed(self, texts):
            return []

    s = Store(tmp_path / "x.db")
    with pytest.raises(StoreError):
        _ingest(s, Bad())
    s.close()


def test_get_chunk_and_stats(tmp_path):
    s = Store(tmp_path / "x.db")
    stats = _ingest(s, HashingEmbedder(dim=64))
    chunk_id = f"ar-x:0000"
    got = s.get_chunk(chunk_id)
    assert got and got.title == "جامعة الجزيرة"
    assert s.get_chunk("missing") is None
    assert stats.chunks == 1
    s.close()
