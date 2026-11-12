"""Ingest the synthetic corpus (16 markdown + 4 PDF documents) into Postgres.

Idempotent: documents whose content did not change are skipped ("unchanged").

Usage:
  uv run --extra local python scripts/ingest_corpus.py                  # free local embeddings
  uv run python scripts/ingest_corpus.py --collection harbor-goods      # Voyage if the key is set
"""

from __future__ import annotations

import argparse

from common import COLLECTION, ingest_corpus, live_settings

from app.embeddings import build_embedder
from app.store import Store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()
    settings = live_settings()
    store = Store(settings.database_url, settings.embedding_dim)
    store.init_schema()
    embedder = build_embedder(settings)
    print(f"Ingesting into {args.collection!r} with {embedder.model} ...")
    ingest_corpus(store, embedder, args.collection, settings)
    print(f"{store.count_chunks(args.collection)} chunks in {args.collection!r}")


if __name__ == "__main__":
    main()
