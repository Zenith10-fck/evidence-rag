"""Settings read from the environment and `.env`.

`get_settings()` is the only entry point. `Settings.check()` fails early with a clear message when a
value needed for the chosen mode is missing, so the app never starts half-configured.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """A required setting is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Mode: live = Postgres + embeddings (+ Claude for /ask); demo = replay recorded answers.
    evidence_rag_mode: Literal["live", "demo"] = "live"

    # Secrets (never logged)
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    api_key: str = ""
    database_url: str = ""

    # Claude (named ANSWER_* so they never collide with CLAUDE_* variables set by other tools)
    answer_model: str = "claude-sonnet-5-5"
    answer_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    answer_max_tokens: int = 8000
    answer_refusal_fallback: bool = True
    # Set explicitly so a stray ANTHROPIC_BASE_URL in the shell cannot redirect API calls.
    answer_api_base_url: str = "https://api.anthropic.com"

    # Embeddings: auto = Voyage API if VOYAGE_API_KEY is set, else the local open-weights model.
    embedding_provider: Literal["auto", "voyage", "local"] = "auto"
    voyage_embedding_model: str = "voyage-4"
    # Local default: BAAI/bge-large-en-v1.5 (MIT, 1024 dims). Weights only (safetensors), no remote
    # code. Revision pinned. Voyage's open-weights MongoDB/voyage-4-nano also works but needs
    # LOCAL_EMBEDDING_TRUST_REMOTE_CODE=true (it runs code from its model repo); see README.
    local_embedding_model: str = "BAAI/bge-large-en-v1.5"
    local_embedding_revision: str = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    local_embedding_trust_remote_code: bool = False
    local_query_prompt: str = "Represent this sentence for searching relevant passages: "
    embedding_dim: int = 1024

    # Reranking (hybrid_rerank mode)
    rerank_model: str = "rerank-3"
    # Voyage client-side rate limits (0 = none). Accounts without a payment method: 3 and 10000.
    voyage_rpm: int = 0
    voyage_tpm: int = 0
    voyage_query_cache: str = ".cache/voyage_queries.json"
    rerank_candidates: int = 20

    # Server-side limits (the caller cannot change these)
    retrieval_k: int = Field(default=5, ge=1, le=20)
    max_question_chars: int = 500
    max_pdf_mb: int = 10
    max_pdf_pages: int = 50
    chunk_max_chars: int = 700

    # Callers. API_KEY is one caller named "default". API_KEYS adds named callers, each with its
    # own key and its own limits: "acme:<key>,beta:<key>". Names show in logs; keys never do.
    api_keys: str = ""
    # Per-caller limits on the routes that cost money (/ask, /retrieve). 0 = no limit.
    # In-memory, per process: with several workers or hosts, put the limit in a gateway or Redis.
    ask_per_minute: int = Field(default=20, ge=0)
    ask_per_day: int = Field(default=500, ge=0)
    # Demo mode has no keys, so the limit is per client IP.
    demo_per_minute: int = Field(default=30, ge=0)

    # Logs: json (one object per line, for a log pipeline) or text (for a terminal).
    log_format: Literal["json", "text"] = "json"
    log_level: str = "INFO"
    # Browser origins allowed to call the API from another site, comma separated. Empty = none.
    cors_origins: str = ""

    # Demo mode files
    corpus_dir: str = "data/corpus"
    recordings_path: str = "data/recorded/answers.json"
    demo_collection: str = "harbor-goods"

    @property
    def is_demo(self) -> bool:
        return self.evidence_rag_mode == "demo"

    @property
    def callers(self) -> dict[str, str]:
        """Caller name -> key, from API_KEY ("default") and API_KEYS. Raises ConfigError on a
        malformed API_KEYS so a typo never silently locks a client out (or lets one in)."""
        out: dict[str, str] = {}
        if self.api_key:
            out["default"] = self.api_key
        for item in filter(None, (p.strip() for p in self.api_keys.split(","))):
            name, sep, key = item.partition(":")
            name, key = name.strip(), key.strip()
            if not sep or not name or not key:
                raise ConfigError("API_KEYS must look like 'name:key,name2:key2'")
            if not name.replace("-", "").replace("_", "").isalnum():
                raise ConfigError(
                    f"API_KEYS: caller name {name!r} may use letters, digits, - and _"
                )
            if name in out:
                raise ConfigError(f"API_KEYS: caller name {name!r} is used twice")
            if len(key) < 16:
                raise ConfigError(f"API_KEYS: the key for {name!r} is shorter than 16 characters")
            out[name] = key
        if len(set(out.values())) != len(out):
            raise ConfigError("API_KEY / API_KEYS: two callers share the same key")
        return out

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_embedding_provider(self) -> Literal["voyage", "local"]:
        if self.embedding_provider == "auto":
            return "voyage" if self.voyage_api_key else "local"
        return self.embedding_provider

    def check(self) -> None:
        """Fail early, with a message that says what to set."""
        if self.is_demo:
            return
        missing = []
        if not self.database_url:
            missing.append("DATABASE_URL")
        if not self.api_key and not self.api_keys:
            missing.append("API_KEY (or API_KEYS)")
        if self.embedding_provider == "voyage" and not self.voyage_api_key:
            missing.append("VOYAGE_API_KEY (EMBEDDING_PROVIDER=voyage)")
        if missing:
            raise ConfigError(
                "Missing settings for live mode: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them, or set EVIDENCE_RAG_MODE=demo."
            )
        self.callers  # noqa: B018 - validates API_KEYS at startup
        if "<POSTGRES_PASSWORD>" in self.database_url:
            raise ConfigError("DATABASE_URL still contains the <POSTGRES_PASSWORD> placeholder.")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.check()
    return settings
