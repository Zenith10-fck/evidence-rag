"""Retrieval-only evaluation: recall@k and MRR per retrieval mode (no Claude calls).

Cost: free with the local embedding model (EMBEDDING_PROVIDER=local, the default without a
VOYAGE_API_KEY). With the Voyage API it costs a few embedding calls; `hybrid_rerank` needs
VOYAGE_API_KEY (reranker) and is reported as not measured without it.

Writes the "retrieval_eval" block of results.json (one entry per embedding model).

Usage:
  uv run --extra local python scripts/eval_retrieval.py          # local model, free
  uv run python scripts/eval_retrieval.py                        # Voyage API if the key is set
"""

from __future__ import annotations

import time

from common import live_settings, questions, read_results, setup, today, write_results

from app.embeddings import voyage_limiter
from app.evaluation import retrieval_metrics
from app.rerank import VoyageReranker
from app.retrieve import retrieve

MODES = ("vector", "hybrid", "hybrid_rerank")


def main() -> None:
    settings = live_settings()
    store, embedder, collection = setup(settings)
    reranker = (
        VoyageReranker(
            settings.voyage_api_key,
            settings.rerank_model,
            voyage_limiter(settings.voyage_api_key, settings.voyage_rpm, settings.voyage_tpm),
        )
        if settings.voyage_api_key
        else None
    )
    qs = questions()
    answerable = [q for q in qs if q["type"] != "unanswerable"]
    block: dict = {
        "date": today(),
        "embedding_model": embedder.model,
        "rerank_model": settings.rerank_model if reranker else None,
        "k": settings.retrieval_k,
        "questions": len(answerable),
        "modes": {},
    }
    for mode in MODES:
        if mode == "hybrid_rerank" and reranker is None:
            block["modes"][mode] = {"not_measured": "no VOYAGE_API_KEY (the reranker is an API)"}
            print(f"{mode:14} not measured (no VOYAGE_API_KEY)")
            continue
        retrieved: dict[str, list[str]] = {}
        skipped = 0
        started = time.perf_counter()
        for q in answerable:
            res = retrieve(
                store,
                embedder,
                collection_id=collection,
                question=q["question"],
                mode=mode,
                k=settings.retrieval_k,
                candidates=settings.rerank_candidates,
                reranker=reranker,
            )
            skipped += res.rerank == "skipped"
            retrieved[q["id"]] = [c.doc_id for c in res.chunks]
        m = retrieval_metrics(qs, retrieved)
        m["seconds"] = round(time.perf_counter() - started, 1)
        if mode == "hybrid_rerank":
            m["rerank_skipped"] = skipped
        block["modes"][mode] = m
        print(
            f"{mode:14} recall@{settings.retrieval_k}={m['recall_at_k']:.3f}  "
            f"hit@1={m['hit_at_1']:.3f}  mrr={m['mrr']:.3f}  "
            f"exact-term recall={m['by_tag']['exact-term']['recall_at_k']}  "
            f"paraphrase recall={m['by_tag']['paraphrase']['recall_at_k']}  misses={m['misses']}"
        )

    results = read_results()
    results.setdefault("retrieval_eval", {})[embedder.model] = block
    write_results(results)


if __name__ == "__main__":
    main()
