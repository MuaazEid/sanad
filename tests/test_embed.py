import io
import json
import urllib.error

import pytest

from sanad.embed import (EmbeddingError, HashingEmbedder, OllamaEmbedder,
                         cosine, get_embedder, pack, unpack)


def test_hashing_is_deterministic_across_instances():
    a = HashingEmbedder(dim=128).embed(["الرياض عاصمة السعودية"])[0]
    b = HashingEmbedder(dim=128).embed(["الرياض عاصمة السعودية"])[0]
    assert a == b


def test_hashing_vectors_are_unit_length():
    vec = HashingEmbedder(dim=64).embed(["مترو الرياض"])[0]
    assert cosine(vec, vec) == pytest.approx(1.0, abs=1e-6)


def test_hashing_prefers_related_text():
    emb = HashingEmbedder(dim=1024)
    query, related, unrelated = emb.embed([
        "مترو الرياض", "قطار الرياض الكهربائي", "تحلية مياه البحر"])
    assert cosine(query, related) > cosine(query, unrelated)


def test_pack_unpack_roundtrip():
    vec = [0.1, -0.2, 0.3]
    assert unpack(pack(vec), 3) == pytest.approx(vec, abs=1e-6)


def test_unpack_rejects_wrong_dimension():
    with pytest.raises(EmbeddingError):
        unpack(pack([0.1, 0.2]), 3)


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])


def test_get_embedder_unknown():
    with pytest.raises(ValueError):
        get_embedder("bogus")


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_embedder_request_and_normalization(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        return _FakeResponse({"embeddings": [[3.0, 4.0], [1.0, 0.0]]})

    monkeypatch.setattr("sanad.embed.urllib.request.urlopen", fake_urlopen)
    emb = OllamaEmbedder(model="bge-m3", batch_size=8)
    vectors = emb.embed(["a", "b"])
    assert seen["url"].endswith("/api/embed")
    assert seen["body"] == {"model": "bge-m3", "input": ["a", "b"]}
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert emb.name == "ollama:bge-m3"


def test_ollama_embedder_http_error_is_actionable(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 404, "not found", {}, io.BytesIO(b'{"error":"model not found"}'))

    monkeypatch.setattr("sanad.embed.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(EmbeddingError, match="ollama pull"):
        OllamaEmbedder().embed(["x"])


def test_ollama_embedder_rejects_bad_shape(monkeypatch):
    monkeypatch.setattr("sanad.embed.urllib.request.urlopen",
                        lambda req, timeout: _FakeResponse({"embeddings": []}))
    with pytest.raises(EmbeddingError, match="unexpected embed response"):
        OllamaEmbedder().embed(["x"])
