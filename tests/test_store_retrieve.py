"""Ingest and retrieval against the real Postgres + pgvector (fake embeddings)."""

from __future__ import annotations

import pytest

from app.corpus import load_corpus
from app.ingest import ingest_document
from app.rerank import RerankError
from app.retrieve import query_terms, retrieve, rrf_merge
from app.store import StoredChunk
from tests.conftest import ROOT

pytestmark = pytest.mark.db

RETURNS_TEXT = (
    "# Returns Policy\n\nCustomers may return unused items within 30 days of delivery. "
    "Return shipping is paid by the customer unless the item arrived damaged. "
    "Refunds are issued to the original payment method."
)


def _ingest(store, emb, cid, doc_id="returns-policy", text=RETURNS_TEXT, title="Returns Policy"):
    return ingest_document(
        store, emb, collection_id=cid, doc_id=doc_id, title=title, version="2026-01",
        source_type="text", pages=[(None, text)], raw_content=text.encode(),
    )


def test_ingest_is_idempotent_by_hash(store, fake_embedder, collection):
    first = _ingest(store, fake_embedder, collection)
    calls = fake_embedder.calls
    second = _ingest(store, fake_embedder, collection)
    assert first.status == "created" and second.status == "unchanged"
    assert fake_embedder.calls == calls  # no embedding call for an unchanged document
    assert store.count_chunks(collection) == first.chunks


def test_changed_document_replaces_its_chunks(store, fake_embedder, collection):
    _ingest(store, fake_embedder, collection)
    res = _ingest(store, fake_embedder, collection, text="# Returns Policy\n\nOnly one line now.")
    assert res.status == "updated"
    assert store.count_chunks(collection) == 1


def test_collection_a_never_sees_collection_b(store, fake_embedder, collection):
    other = collection + "_b"
    try:
        _ingest(store, fake_embedder, collection)
        _ingest(store, fake_embedder, other, doc_id="secret", text="Secret pricing 99 USD for B only.")
        for mode in ("vector", "hybrid", "hybrid_rerank"):
            res = retrieve(
                store, fake_embedder, collection_id=collection,
                question="secret pricing for B", mode=mode, k=5,
            )
            assert all(c.chunk_id.startswith(collection + ":") for c in res.chunks)
            assert all(c.doc_id != "secret" for c in res.chunks)
    finally:
        store.delete_collection(other)


def test_pdf_chunks_keep_page_numbers(store, fake_embedder, collection):
    doc = next(d for d in load_corpus(ROOT / "data" / "corpus") if d.doc_id == "warranty")
    res = ingest_document(
        store, fake_embedder, collection_id=collection, doc_id=doc.doc_id, title=doc.title,
        version=doc.version, source_type="pdf", pages=doc.pages, raw_content=doc.raw_content,
    )
    assert res.pages == 3
    rows = store.list_documents(collection)
    assert rows[0]["source_type"] == "pdf" and rows[0]["pages"] == 3
    hits = retrieve(store, fake_embedder, collection_id=collection, question="UV damage tent fabrics", mode="hybrid", k=3)
    assert hits.chunks[0].page == 2


def test_hybrid_finds_exact_terms(store, fake_embedder, collection):
    _ingest(store, fake_embedder, collection)
    _ingest(store, fake_embedder, collection, doc_id="po", title="Purchase Orders",
            text="Purchase orders above 2,500 USD need approval from the operations manager.")
    res = retrieve(store, fake_embedder, collection_id=collection, question="approval 2,500", mode="hybrid", k=2)
    assert res.chunks[0].doc_id == "po"


class BrokenReranker:
    model = "broken"

    def rerank(self, query, documents, top_k):
        raise RerankError("down")


class ReverseReranker:
    model = "reverse"

    def rerank(self, query, documents, top_k):
        return list(reversed(range(len(documents))))[:top_k]


def test_rerank_failure_falls_back_to_hybrid(store, fake_embedder, collection):
    _ingest(store, fake_embedder, collection)
    hybrid = retrieve(store, fake_embedder, collection_id=collection, question="return items", mode="hybrid", k=3)
    res = retrieve(store, fake_embedder, collection_id=collection, question="return items",
                   mode="hybrid_rerank", k=3, reranker=BrokenReranker())
    assert res.rerank == "skipped"
    assert [c.chunk_id for c in res.chunks] == [c.chunk_id for c in hybrid.chunks]


def test_rerank_reorders(store, fake_embedder, collection):
    _ingest(store, fake_embedder, collection)
    res = retrieve(store, fake_embedder, collection_id=collection, question="return items",
                   mode="hybrid_rerank", k=3, reranker=ReverseReranker())
    assert res.rerank == "applied"


def test_query_terms_cannot_inject_tsquery_operators():
    assert query_terms("a & b | !c (d) HG-TENT-220") == ["a", "b", "c", "d", "hg", "tent", "220"]


def test_rrf_merge_rewards_agreement():
    def ch(i):
        return StoredChunk(f"c:{i}", "c", f"d{i}", "t", 0, None, "x")

    merged = rrf_merge([ch(1), ch(2), ch(3)], [ch(3), ch(4)])
    assert merged[0][0].chunk_id == "c:3"


class OtherModelEmbedder:
    """Same dimension, different model name: must trigger a re-embed, never 'unchanged'."""

    def __init__(self, inner):
        self.inner = inner
        self.model = "another-model"
        self.dim = inner.dim

    def embed_documents(self, texts):
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


def test_changing_the_embedding_model_re_embeds(store, fake_embedder, collection):
    _ingest(store, fake_embedder, collection)
    res = _ingest(store, OtherModelEmbedder(fake_embedder), collection)
    assert res.status == "updated"
