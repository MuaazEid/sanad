"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion.

RRF is used instead of score normalization because BM25 scores and cosine
similarities live on incomparable scales. RRF only needs ranks, so a change
of embedding model does not require re-tuning a blend weight.
"""
from __future__ import annotations

from dataclasses import dataclass

from .embed import Embedder
from .store import Store, StoredChunk
from .text import tokens


@dataclass(frozen=True)
class Retrieved:
    chunk: StoredChunk
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text


class HybridRetriever:
    def __init__(self, store: Store, embedder: Embedder, *,
                 candidates: int = 30, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, query: str, *, top_k: int = 5, lang: str | None = None,
               mode: str = "hybrid") -> list[Retrieved]:
        if mode not in {"hybrid", "lexical"}:
            raise ValueError(f"unknown mode '{mode}' (expected hybrid|lexical)")
        if not query.strip() or not tokens(query):
            return []
        lexical = self.store.search_bm25(query, limit=self.candidates, lang=lang)

        dense: list[StoredChunk] = []
        if mode == "hybrid":
            query_vec = self.embedder.embed([query])[0]
            dense = self.store.search_dense(
                query_vec, self.embedder.name, limit=self.candidates, lang=lang)

        by_id: dict[str, dict] = {}
        for rank, chunk in enumerate(lexical, start=1):
            entry = by_id.setdefault(chunk.id, {"chunk": chunk, "rrf": 0.0})
            entry["rrf"] += 1.0 / (self.rrf_k + rank)
            entry["bm25"] = chunk.score
            entry["bm25_rank"] = rank
        for rank, chunk in enumerate(dense, start=1):
            entry = by_id.setdefault(chunk.id, {"chunk": chunk, "rrf": 0.0})
            entry["rrf"] += 1.0 / (self.rrf_k + rank)
            entry["dense"] = chunk.score
            entry["dense_rank"] = rank

        ranked = sorted(
            by_id.values(),
            key=lambda e: (e["rrf"], e.get("dense", -1.0)),
            reverse=True,
        )
        return [
            Retrieved(
                chunk=e["chunk"], score=e["rrf"],
                bm25_score=e.get("bm25"), dense_score=e.get("dense"),
                bm25_rank=e.get("bm25_rank"), dense_rank=e.get("dense_rank"),
            )
            for e in ranked[:top_k]
        ]
