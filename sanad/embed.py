"""Embedding backends.

OllamaEmbedder is the real path (bge-m3 is multilingual and handles Arabic).
HashingEmbedder is a dependency-free fallback so tests and air-gapped demos
still exercise the full retrieval code path. It is much weaker semantically
and is never allowed to masquerade as the real one: the model name is stored
next to every vector, and the store refuses to mix models.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from array import array
from typing import Protocol

from .text import tokens


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def pack(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def unpack(blob: bytes, dim: int) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    if len(arr) != dim:
        raise EmbeddingError(f"corrupt vector: expected {dim}, got {len(arr)}")
    return arr.tolist()


class OllamaEmbedder:
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434",
                 timeout: float = 120.0, batch_size: int = 16):
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._request(texts[start:start + self.batch_size]))
        return out

    def _request(self, batch: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise EmbeddingError(
                f"Ollama embed failed ({exc.code}): {detail}. "
                f"Is `{self.model}` pulled? Try: ollama pull {self.model}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise EmbeddingError(
                f"Ollama unreachable at {self.base_url}: {exc}. Is `ollama serve` running?") from exc
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise EmbeddingError(f"unexpected embed response shape: {str(payload)[:200]}")
        return [_l2([float(x) for x in v]) for v in vectors]


class HashingEmbedder:
    """Signed feature hashing over words, word bigrams and character 4-grams.

    Deterministic across processes (blake2b, not Python's salted hash), which
    matters because vectors are persisted and re-read by a later CLI call.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _features(self, text: str) -> list[str]:
        toks = tokens(text)
        feats = [f"w:{t}" for t in toks]
        feats += [f"b:{a}_{b}" for a, b in zip(toks, toks[1:])]
        for t in toks:
            if len(t) >= 4:
                feats += [f"c:{t[i:i + 4]}" for i in range(len(t) - 3)]
        return feats

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            sign = 1.0 if h & 1 else -1.0
            vec[(h >> 1) % self.dim] += sign
        return _l2(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def get_embedder(name: str, *, model: str = "bge-m3",
                 base_url: str = "http://localhost:11434") -> Embedder:
    if name == "ollama":
        return OllamaEmbedder(model=model, base_url=base_url)
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder '{name}' (expected ollama|hashing)")
