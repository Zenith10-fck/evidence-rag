"""Optional Voyage reranker for the `hybrid_rerank` mode.

A reranker reads the question and each candidate chunk together (a cross-encoder), so it can judge
relevance better than comparing two separately computed embeddings. It costs one extra API call per
question, so it is only used in the `hybrid_rerank` mode, on the hybrid top 20.
"""

from __future__ import annotations

from typing import Protocol

from app.ratelimit import RateLimiter, call_with_rate_limit_retry, estimate_tokens


class RerankError(RuntimeError):
    pass


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Indices into `documents`, best first, at most `top_k`."""
        ...


class VoyageReranker:
    def __init__(self, api_key: str, model: str, limiter: RateLimiter | None = None):
        import voyageai

        self._client = voyageai.Client(api_key=api_key, max_retries=0, timeout=30)
        self._rate_error = voyageai.error.RateLimitError
        self.model = model
        self._limiter = limiter or RateLimiter()

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        # Voyage counts query tokens once per document, plus the document tokens.
        tokens = estimate_tokens([query]) * len(documents) + estimate_tokens(documents)

        def call():
            self._limiter.acquire(tokens)
            return self._client.rerank(query, documents, model=self.model, top_k=top_k)

        try:
            result = call_with_rate_limit_retry(call, lambda e: isinstance(e, self._rate_error))
        except Exception as exc:
            raise RerankError(f"Voyage rerank failed: {type(exc).__name__}") from exc
        return [r.index for r in result.results]
