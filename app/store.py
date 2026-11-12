"""Postgres + pgvector reads and writes.

Every query that reads chunks filters by `collection_id` in SQL, so a question in one collection
can never see another collection's chunks.

Vector index: HNSW (cosine). Chosen over IVFFlat because it needs no training step (IVFFlat must
be built after data exists and re-tuned as it grows), and it gives good recall at this size. Because
the collection filter is applied after the index scan, `hnsw.iterative_scan = relaxed_order`
(pgvector 0.8+) is set per session so a filtered query still returns k rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


class StoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    collection_id: str
    doc_id: str
    title: str
    position: int
    page: int | None
    text: str


@dataclass(frozen=True)
class ChunkRow:
    """A chunk ready to be written (embedding already computed)."""

    position: int
    page: int | None
    text: str
    embedding: list[float]


def content_hash(
    title: str, version: str, content: bytes, embedding_model: str = "", chunk_max_chars: int = 0
) -> str:
    """Hash of everything that decides the stored chunks and vectors. A change of embedding model
    or chunk size makes the document 'updated' (re-embedded), never silently 'unchanged'."""
    h = hashlib.sha256()
    h.update(json.dumps([title, version, embedding_model, chunk_max_chars]).encode())
    h.update(content)
    return h.hexdigest()


def make_chunk_id(collection_id: str, doc_id: str, position: int) -> str:
    return f"{collection_id}:{doc_id}:{position}"


def _schema_v1(dim: int) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """CREATE TABLE IF NOT EXISTS documents (
            collection_id text NOT NULL,
            doc_id text NOT NULL,
            title text NOT NULL,
            version text NOT NULL DEFAULT '',
            source_type text NOT NULL CHECK (source_type IN ('text', 'pdf')),
            content_hash text NOT NULL,
            pages int,
            embedding_model text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (collection_id, doc_id)
        )""",
        f"""CREATE TABLE IF NOT EXISTS chunks (
            chunk_id text PRIMARY KEY,
            collection_id text NOT NULL,
            doc_id text NOT NULL,
            position int NOT NULL,
            page int,
            text text NOT NULL,
            embedding vector({dim}) NOT NULL,
            tsv tsvector NOT NULL,
            FOREIGN KEY (collection_id, doc_id)
                REFERENCES documents (collection_id, doc_id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)",
        "CREATE INDEX IF NOT EXISTS chunks_collection_doc ON chunks (collection_id, doc_id)",
    ]


# Ordered, append-only. Never edit a released migration: add the next version instead.
# Each entry is (version, description, function(dim) -> list of SQL statements).
MIGRATIONS = [
    (1, "documents, chunks, vector/full-text indexes", _schema_v1),
]
_MIGRATION_LOCK = 7_433_021  # advisory lock id: two starting instances cannot migrate at once


class Store:
    def __init__(self, database_url: str, dim: int):
        self.database_url = database_url
        self.dim = dim

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        try:
            # autocommit: each `conn.transaction()` block is a real transaction (not a savepoint
            # inside an implicit one that would be rolled back on close).
            conn = psycopg.connect(
                self.database_url, row_factory=dict_row, connect_timeout=5, autocommit=True
            )
        except psycopg.OperationalError as exc:
            raise StoreError(f"cannot connect to Postgres: {exc.__class__.__name__}") from exc
        try:
            register_vector(conn)
            conn.execute("SET hnsw.iterative_scan = relaxed_order")
            yield conn
        finally:
            conn.close()

    def ping(self) -> None:
        """Raise StoreError if the database cannot be reached or is not fully migrated.
        Used by the readiness check, so an orchestrator stops sending traffic to a broken node."""
        with self.connect() as conn:
            row = conn.execute("SELECT max(version) AS v FROM schema_migrations").fetchone()
            if not row or row["v"] != MIGRATIONS[-1][0]:
                raise StoreError("database schema is not up to date")

    def init_schema(self) -> None:
        # The extension must exist before register_vector can find the type.
        try:
            with psycopg.connect(self.database_url, connect_timeout=5, autocommit=True) as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.OperationalError as exc:
            raise StoreError(f"cannot connect to Postgres: {exc.__class__.__name__}") from exc
        with self.connect() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
            conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version int PRIMARY KEY,
                    description text NOT NULL,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )"""
            )
            done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}
            unknown = done - {v for v, _, _ in MIGRATIONS}
            if unknown:
                raise StoreError(
                    f"the database has migration(s) {sorted(unknown)} that this version of the "
                    "app does not know: refusing to start against a newer schema."
                )
            for version, description, statements in MIGRATIONS:
                if version in done:
                    continue
                for stmt in statements(self.dim):
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO schema_migrations (version, description) VALUES (%s, %s)",
                    (version, description),
                )
            row = conn.execute(
                "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
                "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            ).fetchone()
            if row and row["t"] != f"vector({self.dim})":
                raise StoreError(
                    f"chunks.embedding is {row['t']} but EMBEDDING_DIM={self.dim}. "
                    "Use the same dimension or recreate the database volume."
                )

    # ---------- documents ----------

    def get_hash(self, collection_id: str, doc_id: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM documents WHERE collection_id=%s AND doc_id=%s",
                (collection_id, doc_id),
            ).fetchone()
        return row["content_hash"] if row else None

    def replace_document(
        self,
        *,
        collection_id: str,
        doc_id: str,
        title: str,
        version: str,
        source_type: str,
        content_hash: str,
        pages: int | None,
        embedding_model: str,
        chunks: list[ChunkRow],
    ) -> str:
        """Insert or replace a document and all its chunks in one transaction.

        Returns 'created' or 'updated'.
        """
        with self.connect() as conn, conn.transaction():
            # Serialise concurrent writes of the same document (a new document has no row to lock).
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{collection_id}:{doc_id}",)
            )
            existed = conn.execute(
                "SELECT 1 FROM documents WHERE collection_id=%s AND doc_id=%s FOR UPDATE",
                (collection_id, doc_id),
            ).fetchone()
            conn.execute(
                "DELETE FROM chunks WHERE collection_id=%s AND doc_id=%s", (collection_id, doc_id)
            )
            conn.execute(
                """INSERT INTO documents (collection_id, doc_id, title, version, source_type,
                       content_hash, pages, embedding_model)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (collection_id, doc_id) DO UPDATE SET
                       title=EXCLUDED.title, version=EXCLUDED.version,
                       source_type=EXCLUDED.source_type, content_hash=EXCLUDED.content_hash,
                       pages=EXCLUDED.pages, embedding_model=EXCLUDED.embedding_model,
                       updated_at=now()""",
                (
                    collection_id,
                    doc_id,
                    title,
                    version,
                    source_type,
                    content_hash,
                    pages,
                    embedding_model,
                ),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (chunk_id, collection_id, doc_id, position, page, text,
                           embedding, tsv)
                       VALUES (%s, %s, %s, %s, %s, %s, %s,
                               to_tsvector('english', %s))""",
                    [
                        (
                            make_chunk_id(collection_id, doc_id, c.position),
                            collection_id,
                            doc_id,
                            c.position,
                            c.page,
                            c.text,
                            np.asarray(c.embedding, dtype=np.float32),
                            f"{title}\n{c.text}",
                        )
                        for c in chunks
                    ],
                )
        return "updated" if existed else "created"

    def list_documents(self, collection_id: str | None) -> list[dict]:
        sql = """SELECT d.collection_id, d.doc_id, d.title, d.version, d.source_type, d.pages,
                        count(c.chunk_id)::int AS chunks
                 FROM documents d
                 LEFT JOIN chunks c ON c.collection_id=d.collection_id AND c.doc_id=d.doc_id
                 {where}
                 GROUP BY d.collection_id, d.doc_id
                 ORDER BY d.collection_id, d.doc_id"""
        with self.connect() as conn:
            if collection_id:
                rows = conn.execute(
                    sql.format(where="WHERE d.collection_id=%s"), (collection_id,)
                ).fetchall()
            else:
                rows = conn.execute(sql.format(where="")).fetchall()
        return rows

    def count_chunks(self, collection_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT count(*)::int AS n FROM chunks WHERE collection_id=%s", (collection_id,)
            ).fetchone()
        return row["n"]

    def delete_collection(self, collection_id: str) -> None:
        with self.connect() as conn, conn.transaction():
            conn.execute("DELETE FROM documents WHERE collection_id=%s", (collection_id,))

    # ---------- search ----------

    def vector_search(
        self, collection_id: str, query_vec: list[float], limit: int
    ) -> list[tuple[StoredChunk, float]]:
        """Nearest chunks by cosine distance. Score = 1 - distance."""
        with self.connect() as conn:
            # iterative_scan=relaxed_order may return rows slightly out of order; the materialized
            # CTE plus an outer ORDER BY restores exact distance order (pgvector docs).
            rows = conn.execute(
                """WITH hits AS MATERIALIZED (
                       SELECT c.chunk_id, c.collection_id, c.doc_id, d.title, c.position, c.page,
                              c.text, c.embedding <=> %s AS distance
                       FROM chunks c
                       JOIN documents d
                         ON d.collection_id=c.collection_id AND d.doc_id=c.doc_id
                       WHERE c.collection_id=%s
                       ORDER BY c.embedding <=> %s
                       LIMIT %s
                   )
                   SELECT * FROM hits ORDER BY distance, chunk_id""",
                (
                    np.asarray(query_vec, dtype=np.float32),
                    collection_id,
                    np.asarray(query_vec, dtype=np.float32),
                    limit,
                ),
            ).fetchall()
        return [(_to_chunk(r), 1.0 - float(r["distance"])) for r in rows]

    def text_search(
        self, collection_id: str, terms: list[str], limit: int
    ) -> list[tuple[StoredChunk, float]]:
        """Full-text search: any of the terms (OR), ranked by ts_rank_cd."""
        if not terms:
            return []
        tsquery = " | ".join(terms)
        with self.connect() as conn:
            rows = conn.execute(
                """WITH q AS (SELECT to_tsquery('english', %s) AS query)
                   SELECT c.chunk_id, c.collection_id, c.doc_id, d.title, c.position, c.page,
                          c.text, ts_rank_cd(c.tsv, q.query) AS rank
                   FROM chunks c
                   JOIN documents d ON d.collection_id=c.collection_id AND d.doc_id=c.doc_id,
                        q
                   WHERE c.collection_id=%s AND c.tsv @@ q.query
                   ORDER BY rank DESC, c.chunk_id
                   LIMIT %s""",
                (tsquery, collection_id, limit),
            ).fetchall()
        return [(_to_chunk(r), float(r["rank"])) for r in rows]


def _to_chunk(r: dict) -> StoredChunk:
    return StoredChunk(
        chunk_id=r["chunk_id"],
        collection_id=r["collection_id"],
        doc_id=r["doc_id"],
        title=r["title"],
        position=r["position"],
        page=r["page"],
        text=r["text"],
    )
