"""Corpus manifest loading and ingestion."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .embed import Embedder
from .store import IngestStats, Store
from .text import Chunk, chunk_text


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    lang: str
    path: Path
    source_url: str
    license: str
    revision: str

    @property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")


def load_manifest(corpus_dir: str | Path) -> list[Document]:
    root = Path(corpus_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs: list[Document] = []
    for item in data.get("documents", []):
        path = root.parent / item["path"] if not Path(item["path"]).is_absolute() else Path(item["path"])
        docs.append(Document(
            id=item["id"], title=item["title"], lang=item["lang"], path=path,
            source_url=item.get("source_url", ""), license=item.get("license", ""),
            revision=str(item.get("revision", "")),
        ))
    if not docs:
        raise ValueError("manifest contains no documents")
    return docs


def ingest_corpus(store: Store, embedder: Embedder, corpus_dir: str | Path,
                  *, force: bool = False, max_chars: int = 900) -> list[IngestStats]:
    results: list[IngestStats] = []
    for doc in load_manifest(corpus_dir):
        text = doc.text
        chunks: list[Chunk] = chunk_text(text, max_chars=max_chars)
        stats = store.ingest(
            doc_id=doc.id, title=doc.title, lang=doc.lang, text=text,
            chunks=chunks, embedder=embedder, source_url=doc.source_url,
            license=doc.license, revision=doc.revision, force=force)
        results.append(stats)
    return results
