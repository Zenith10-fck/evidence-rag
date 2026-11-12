"""Shared helpers for the scripts: settings, corpus ingest, results.json read/merge/write."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.corpus import load_corpus, load_questions  # noqa: E402
from app.embeddings import Embedder, build_embedder  # noqa: E402
from app.ingest import ingest_document  # noqa: E402
from app.store import Store  # noqa: E402

COLLECTION = "harbor-goods"
RESULTS = ROOT / "results.json"
QUESTIONS = ROOT / "data" / "questions" / "questions.json"


def live_settings() -> Settings:
    settings = Settings(_env_file=str(ROOT / ".env"))
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set. Copy .env.example to .env and start the db.")
    return settings


def collection_for(embedder: Embedder) -> str:
    """One collection per embedding model, so vectors from different models never mix."""
    safe = "".join(ch if ch.isalnum() else "-" for ch in embedder.model.split("/")[-1])
    return f"{COLLECTION}__{safe}"[:64]


def ingest_corpus(store: Store, embedder: Embedder, collection: str, settings: Settings) -> None:
    docs = load_corpus(ROOT / settings.corpus_dir, settings.max_pdf_pages)
    for doc in docs:
        res = ingest_document(
            store,
            embedder,
            collection_id=collection,
            doc_id=doc.doc_id,
            title=doc.title,
            version=doc.version,
            source_type=doc.source_type,
            pages=doc.pages,
            raw_content=doc.raw_content,
            chunk_max_chars=settings.chunk_max_chars,
        )
        print(f"  {res.status:9} {doc.doc_id:28} chunks={res.chunks}")


def setup(settings: Settings) -> tuple[Store, Embedder, str]:
    store = Store(settings.database_url, settings.embedding_dim)
    store.init_schema()
    embedder = build_embedder(settings)
    collection = collection_for(embedder)
    print(f"Ingesting corpus into collection {collection!r} with {embedder.model} ...")
    ingest_corpus(store, embedder, collection, settings)
    return store, embedder, collection


def questions() -> list[dict]:
    return load_questions(QUESTIONS)


def read_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text("utf-8"))
    return {}


def write_results(data: dict) -> None:
    data.setdefault("note", "Written by scripts only. Never edited by hand.")
    RESULTS.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {RESULTS.name}")


def today() -> str:
    return date.today().isoformat()
