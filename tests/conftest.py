"""Shared fixtures. No test calls a paid API: embeddings and Claude are replaced by fakes.

Database tests use the Postgres from docker compose (or CI's service) and are skipped when it is
unreachable. They write to unique collection ids and delete them afterwards.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

import pytest

from app.answer import LLMResult
from app.config import Settings
from app.store import Store, StoreError

ROOT = Path(__file__).resolve().parent.parent
DIM = 1024


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: same words -> similar vectors."""

    model = "fake-hash-embedder"
    dim = DIM

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)  # noqa: S324 - not security
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vec(text)


class FakeLLM:
    """Returns scripted responses in order; records the prompts it was given."""

    model = "fake-llm"

    def __init__(self, responses: list[str | dict | LLMResult]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> LLMResult:
        self.prompts.append(user)
        r = self.responses.pop(0)
        if isinstance(r, LLMResult):
            return r
        text = r if isinstance(r, str) else json.dumps(r)
        return LLMResult(text=text, stop_reason="end_turn", input_tokens=100, output_tokens=50)


def make_settings(**overrides) -> Settings:
    base = dict(
        evidence_rag_mode="live",
        api_key="test-key",
        database_url=os.environ.get("TEST_DATABASE_URL") or _env_database_url(),
        anthropic_api_key="",
        voyage_api_key="",
        corpus_dir=str(ROOT / "data" / "corpus"),
        recordings_path=str(ROOT / "data" / "recorded" / "answers.json"),
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _env_database_url() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text("utf-8").splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    return "postgresql://evidence:local-dev-only@127.0.0.1:5433/evidence_rag"


@pytest.fixture(scope="session")
def store() -> Store:
    s = Store(make_settings().database_url, DIM)
    try:
        s.init_schema()
    except StoreError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    return s


@pytest.fixture
def collection(store: Store):
    cid = f"test_{uuid.uuid4().hex[:12]}"
    yield cid
    store.delete_collection(cid)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(scope="session")
def questions() -> list[dict]:
    return json.loads((ROOT / "data" / "questions" / "questions.json").read_text("utf-8"))
