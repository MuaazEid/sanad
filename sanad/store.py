"""SQLite persistence: documents, chunks, FTS5 index and vectors.

One file holds everything the system needs to answer offline. FTS5 gives
real BM25 instead of a hand-rolled TF-IDF, and the dense vectors live beside
it so a hybrid query is one process, one file, no vector database to operate.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .embed import Embedder, pack, unpack
from .text import Chunk, content_tokens, normalize, tokens

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    lang TEXT NOT NULL,
    source_url TEXT DEFAULT '',
    license TEXT DEFAULT '',
    revision TEXT DEFAULT '',
    content_hash TEXT NOT NULL,
    n_chars INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    section TEXT DEFAULT '',
    text TEXT NOT NULL,
    norm_text TEXT NOT NULL,
    n_tokens INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    norm_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class StoreError(RuntimeError):
    pass


class EmbeddingModelMismatch(StoreError):
    pass


@dataclass(frozen=True)
class StoredChunk:
    id: str
    doc_id: str
    title: str
    lang: str
    section: str
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class IngestStats:
    doc_id: str
    title: str
    chunks: int
    skipped: bool = False


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── meta ───────────────────────────────────────────────────────
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.conn.commit()

    # ── ingest ─────────────────────────────────────────────────────
    def ingest(self, *, doc_id: str, title: str, lang: str, text: str,
               chunks: list[Chunk], embedder: Embedder, source_url: str = "",
               license: str = "", revision: str = "", force: bool = False) -> IngestStats:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self.conn.execute(
            "SELECT content_hash FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if existing and existing["content_hash"] == digest and not force:
            n = self.conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()["n"]
            return IngestStats(doc_id, title, n, skipped=True)

        indexed_with = self.get_meta("embedding_model")
        if indexed_with and indexed_with != embedder.name:
            count = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            if count:
                raise EmbeddingModelMismatch(
                    f"index was built with '{indexed_with}', current embedder is "
                    f"'{embedder.name}'. Re-ingest with --reset or use the same embedder.")

        vectors = embedder.embed([c.text for c in chunks]) if chunks else []
        if len(vectors) != len(chunks):
            raise StoreError("embedder returned a different number of vectors than chunks")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.conn:
            old_rows = self.conn.execute(
                "SELECT rowid FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
            for row in old_rows:
                self.conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (row["rowid"],))
            self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self.conn.execute(
                "INSERT INTO documents(id, title, lang, source_url, license, revision,"
                " content_hash, n_chars, ingested_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (doc_id, title, lang, source_url, license, revision, digest, len(text), now))
            for chunk, vec in zip(chunks, vectors):
                cur = self.conn.execute(
                    "INSERT INTO chunks(id, doc_id, ordinal, section, text, norm_text, n_tokens)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (f"{doc_id}:{chunk.ordinal:04d}", doc_id, chunk.ordinal, chunk.section,
                     chunk.text, normalize(chunk.text), chunk.n_tokens))
                rowid = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid, norm_text) VALUES (?, ?)",
                    (rowid, normalize(chunk.text)))
                self.conn.execute(
                    "INSERT INTO embeddings(chunk_id, model, dim, vector) VALUES (?,?,?,?)",
                    (f"{doc_id}:{chunk.ordinal:04d}", embedder.name, len(vec), pack(vec)))
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('embedding_model', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (embedder.name,))
        return IngestStats(doc_id, title, len(chunks))

    # ── search ─────────────────────────────────────────────────────
    def search_bm25(self, query: str, limit: int = 20, lang: str | None = None) -> list[StoredChunk]:
        toks = list(dict.fromkeys(tokens(query)))
        if not toks:
            return []
        match = " OR ".join(f'"{t}"' for t in toks)
        sql = (
            "SELECT c.id, c.doc_id, d.title, d.lang, c.section, c.text,"
            "       bm25(chunks_fts) AS rank "
            "FROM chunks_fts "
            "JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "JOIN documents d ON d.id = c.doc_id "
            "WHERE chunks_fts MATCH ? "
        )
        params: list[object] = [match]
        if lang:
            sql += "AND d.lang = ? "
            params.append(lang)
        sql += "ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [StoredChunk(r["id"], r["doc_id"], r["title"], r["lang"], r["section"],
                            r["text"], -float(r["rank"])) for r in rows]

    def search_dense(self, query_vec: list[float], model: str, limit: int = 20,
                     lang: str | None = None) -> list[StoredChunk]:
        indexed_with = self.get_meta("embedding_model")
        if indexed_with and indexed_with != model:
            raise EmbeddingModelMismatch(
                f"index vectors come from '{indexed_with}', query used '{model}'")
        sql = (
            "SELECT c.id, c.doc_id, d.title, d.lang, c.section, c.text,"
            "       e.dim, e.vector "
            "FROM embeddings e "
            "JOIN chunks c ON c.id = e.chunk_id "
            "JOIN documents d ON d.id = c.doc_id "
            "WHERE e.model = ? "
        )
        params: list[object] = [model]
        if lang:
            sql += "AND d.lang = ? "
            params.append(lang)
        scored = []
        for row in self.conn.execute(sql, params):
            vec = unpack(row["vector"], row["dim"])
            scored.append(StoredChunk(row["id"], row["doc_id"], row["title"], row["lang"],
                                      row["section"], row["text"],
                                      sum(a * b for a, b in zip(vec, query_vec))))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:limit]

    # ── introspection ──────────────────────────────────────────────
    def stats(self) -> dict:
        q = self.conn.execute
        docs = q("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = q("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        vectors = q("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]
        by_lang = {r["lang"]: r["n"] for r in
                   q("SELECT lang, COUNT(*) AS n FROM documents GROUP BY lang")}
        return {"documents": docs, "chunks": chunks, "vectors": vectors,
                "by_lang": by_lang, "embedding_model": self.get_meta("embedding_model")}

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        row = self.conn.execute(
            "SELECT c.id, c.doc_id, d.title, d.lang, c.section, c.text "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
            (chunk_id,)).fetchone()
        return StoredChunk(**dict(row)) if row else None

    def export_json(self) -> str:
        return json.dumps(self.stats(), ensure_ascii=False)
