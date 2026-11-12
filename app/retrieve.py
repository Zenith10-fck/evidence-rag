"""Retrieval in three modes.

- `vector`: nearest chunks by embedding (cosine).
- `hybrid`: vector top N and Postgres full-text top N, merged with reciprocal rank fusion (RRF).
  Full-text catches exact terms (codes, rates, limits) that embeddings blur.
- `hybrid_rerank`: the hybrid top N reordered by the Voyage reranker. If the reranker is missing or
  fails, the hybrid order is used and `rerank` is reported as `skipped` (the question never fails
  because of the reranker).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.embeddings import Embedder
from app.rerank import Reranker, RerankError
from app.store import Store, StoredChunk

RRF_K = 60
_TERM = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    position: int
    page: int | None
    text: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    rerank: str | None  # applied | skipped | None (not a rerank mode)


def query_terms(question: str) -> list[str]:
    """Safe tokens for to_tsquery (letters and digits only; no operators can be injected)."""
    seen: list[str] = []
    for term in _TERM.findall(question.lower()):
        if term not in seen:
            seen.append(term)
    return seen


def rrf_merge(*ranked_lists: list[StoredChunk], k: int = RRF_K) -> list[tuple[StoredChunk, float]]:
    scores: dict[str, float] = {}
    by_id: dict[str, StoredChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.chunk_id] = chunk
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [(by_id[cid], scores[cid]) for cid in order]


def _out(chunk: StoredChunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        position=chunk.position,
        page=chunk.page,
        text=chunk.text,
        score=round(score, 6),
    )


def retrieve(
    store: Store,
    embedder: Embedder,
    *,
    collection_id: str,
    question: str,
    mode: str,
    k: int,
    candidates: int = 20,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    qvec = embedder.embed_query(question)
    if mode == "vector":
        hits = store.vector_search(collection_id, qvec, k)
        return RetrievalResult([_out(c, s) for c, s in hits], None)

    vec_hits = [c for c, _ in store.vector_search(collection_id, qvec, candidates)]
    txt_hits = [c for c, _ in store.text_search(collection_id, query_terms(question), candidates)]
    fused = rrf_merge(vec_hits, txt_hits)[:candidates]
    if mode == "hybrid":
        return RetrievalResult([_out(c, s) for c, s in fused[:k]], None)

    if mode != "hybrid_rerank":
        raise ValueError(f"unknown mode {mode}")
    if reranker is None or not fused:
        return RetrievalResult([_out(c, s) for c, s in fused[:k]], "skipped")
    try:
        order = reranker.rerank(question, [f"{c.title}\n{c.text}" for c, _ in fused], top_k=k)
    except RerankError:
        return RetrievalResult([_out(c, s) for c, s in fused[:k]], "skipped")
    reranked = [fused[i] for i in order if 0 <= i < len(fused)][:k]
    # score after rerank = its new rank position, best = 1.0
    return RetrievalResult(
        [_out(c, 1.0 / (pos + 1)) for pos, (c, _) in enumerate(reranked)], "applied"
    )
