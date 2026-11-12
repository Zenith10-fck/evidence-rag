"""Ingestion: pages -> chunks -> embeddings -> Postgres, idempotent by content hash.

Embeddings are computed BEFORE the database transaction, so when the embedding service is down
nothing is stored (the caller gets 503) and the previous version of the document stays intact.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunking import chunk_pages
from app.embeddings import Embedder
from app.store import ChunkRow, Store, content_hash


@dataclass(frozen=True)
class IngestResult:
    collection_id: str
    doc_id: str
    status: str  # created | updated | unchanged
    chunks: int
    pages: int | None


def embedding_input(title: str, text: str) -> str:
    """What is embedded for a chunk: the document title gives the chunk its context."""
    return f"{title}\n{text}"


def ingest_document(
    store: Store,
    embedder: Embedder,
    *,
    collection_id: str,
    doc_id: str,
    title: str,
    version: str,
    source_type: str,
    pages: list[tuple[int | None, str]],
    raw_content: bytes,
    chunk_max_chars: int = 700,
) -> IngestResult:
    digest = content_hash(title, version, raw_content, embedder.model, chunk_max_chars)
    page_count = len(pages) if source_type == "pdf" else None
    chunks = chunk_pages(pages, max_chars=chunk_max_chars)
    if store.get_hash(collection_id, doc_id) == digest:
        return IngestResult(collection_id, doc_id, "unchanged", len(chunks), page_count)
    if not chunks:
        raise ValueError("document has no text to index")
    vectors = embedder.embed_documents([embedding_input(title, c.text) for c in chunks])
    rows = [
        ChunkRow(position=c.position, page=c.page, text=c.text, embedding=v)
        for c, v in zip(chunks, vectors, strict=True)
    ]
    status = store.replace_document(
        collection_id=collection_id,
        doc_id=doc_id,
        title=title,
        version=version,
        source_type=source_type,
        content_hash=digest,
        pages=page_count,
        embedding_model=embedder.model,
        chunks=rows,
    )
    return IngestResult(collection_id, doc_id, status, len(rows), page_count)
