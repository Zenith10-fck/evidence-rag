"""Embedding providers.

- `VoyageEmbedder`: the Voyage API (default model `voyage-4`, 1024 dimensions), with Voyage's
  `input_type` retrieval prompts for queries and documents.
- `LocalEmbedder`: a sentence-transformers model on the CPU, free, installed with
  `uv sync --extra local`. Default `BAAI/bge-large-en-v1.5` (1024 dimensions, weights only, no
  remote code). `MongoDB/voyage-4-nano` (Voyage's open-weights model, same embedding space as the
  voyage-4 API) can be used by setting LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_REVISION,
  LOCAL_EMBEDDING_TRUST_REMOTE_CODE=true and LOCAL_QUERY_PROMPT= (empty).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.ratelimit import (
    RateLimiter,
    call_with_rate_limit_retry,
    estimate_tokens,
    token_batches,
)


class EmbeddingError(RuntimeError):
    """The embedding service is unreachable or failed. Maps to HTTP 503."""


class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VoyageEmbedder:
    batch_size = 128

    def __init__(
        self,
        api_key: str,
        model: str,
        dim: int,
        limiter: RateLimiter | None = None,
        cache_path: str | None = None,
    ):
        import voyageai

        # SDK retries are off: they would spend the per-minute quota; we retry after a full window.
        self._client = voyageai.Client(api_key=api_key, max_retries=0, timeout=60)
        self._rate_error = voyageai.error.RateLimitError
        self.model = model
        self.dim = dim
        self._limiter = limiter or RateLimiter()
        self._cache = QueryCache(cache_path, f"{model}:{dim}")

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        # Stay under the tokens-per-minute limit with a margin, and under the API batch size.
        max_tokens = int(self._limiter.tpm * 0.8) if self._limiter.tpm else 100_000
        try:
            for batch in token_batches(texts, max_tokens, self.batch_size):
                chunk = [texts[i] for i in batch]
                tokens = estimate_tokens(chunk)

                def call(chunk=chunk, tokens=tokens):
                    self._limiter.acquire(tokens)
                    return self._client.embed(
                        chunk, model=self.model, input_type=input_type, output_dimension=self.dim
                    )

                result = call_with_rate_limit_retry(call, lambda e: isinstance(e, self._rate_error))
                out.extend(result.embeddings)
        except Exception as exc:  # the voyageai client raises several types
            raise EmbeddingError(f"Voyage embedding failed: {type(exc).__name__}") from exc
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = self._embed([text], "query")[0]
        self._cache.put(text, vector)
        return vector


class QueryCache:
    """Query embeddings cached on disk (optional), so repeated evaluation runs do not pay or
    wait for the same query twice. Keyed by model and dimension."""

    def __init__(self, path: str | None, namespace: str):
        self.path = Path(path) if path else None
        self.namespace = namespace
        self._data: dict[str, list[float]] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text("utf-8")).get(namespace, {})
            except (OSError, ValueError):
                self._data = {}

    def get(self, text: str) -> list[float] | None:
        return self._data.get(text)

    def put(self, text: str, vector: list[float]) -> None:
        self._data[text] = vector
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            all_data = json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}
        except (OSError, ValueError):
            all_data = {}
        all_data[self.namespace] = self._data
        self.path.write_text(json.dumps(all_data), encoding="utf-8")


_LIMITERS: dict[tuple[str, int, int], RateLimiter] = {}


def voyage_limiter(api_key: str, rpm: int, tpm: int) -> RateLimiter:
    """One limiter per Voyage account: embeddings and reranking share the same limits."""
    key = (api_key[-8:], rpm, tpm)
    if key not in _LIMITERS:
        _LIMITERS[key] = RateLimiter(rpm=rpm, tpm=tpm)
    return _LIMITERS[key]


class LocalEmbedder:
    def __init__(
        self,
        model: str,
        dim: int,
        revision: str | None = None,
        trust_remote_code: bool = False,
        query_prompt: str = "",
    ):
        self.model = model
        self.dim = dim
        self.query_prompt = query_prompt
        self._st = _load_sentence_transformer(model, dim, revision, trust_remote_code)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._st.encode_document(
            texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=True
        )
        return [v.astype(float).tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        kwargs = {"prompt": self.query_prompt} if self.query_prompt else {}
        vector = self._st.encode_query(
            text, convert_to_numpy=True, normalize_embeddings=True, **kwargs
        )
        return vector.astype(float).tolist()


@lru_cache(maxsize=2)
def _load_sentence_transformer(model: str, dim: int, revision: str | None, trust: bool):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise EmbeddingError(
            "Local embeddings need the optional extra: run `uv sync --extra local`, "
            "or set VOYAGE_API_KEY to use the Voyage API."
        ) from exc
    # Revision pinned; safetensors only (no pickle). trust_remote_code stays off unless the
    # operator turns it on for a model that needs it.
    st = SentenceTransformer(
        model,
        revision=revision,
        trust_remote_code=trust,
        truncate_dim=dim,
        device="cpu",
        model_kwargs={"use_safetensors": True},
    )
    produced = st.get_embedding_dimension()
    if produced != dim:
        raise EmbeddingError(f"{model} produces {produced} dimensions; EMBEDDING_DIM is {dim}")
    return st


def build_embedder(settings: Settings) -> Embedder:
    if settings.resolved_embedding_provider == "voyage":
        return VoyageEmbedder(
            settings.voyage_api_key,
            settings.voyage_embedding_model,
            settings.embedding_dim,
            limiter=voyage_limiter(
                settings.voyage_api_key, settings.voyage_rpm, settings.voyage_tpm
            ),
            cache_path=settings.voyage_query_cache or None,
        )
    return LocalEmbedder(
        settings.local_embedding_model,
        settings.embedding_dim,
        settings.local_embedding_revision,
        settings.local_embedding_trust_remote_code,
        settings.local_query_prompt,
    )
