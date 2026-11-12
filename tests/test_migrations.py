"""Versioned schema migrations. Needs the Postgres test database (skipped if unreachable)."""

from __future__ import annotations

import pytest

from app.store import MIGRATIONS, StoreError

pytestmark = pytest.mark.db


def _versions(store) -> list[int]:
    with store.connect() as conn:
        rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    return [r["version"] for r in rows]


def test_migrations_are_numbered_without_gaps():
    assert [v for v, _, _ in MIGRATIONS] == list(range(1, len(MIGRATIONS) + 1))


def test_every_migration_is_recorded_once(store):
    store.init_schema()
    store.init_schema()  # running again must change nothing
    assert _versions(store) == [v for v, _, _ in MIGRATIONS]


def test_ping_passes_on_a_migrated_database(store):
    store.ping()


def test_a_newer_schema_stops_the_start(store):
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (9999, 'from the future')"
        )
    try:
        with pytest.raises(StoreError, match="does not know"):
            store.init_schema()
    finally:
        with store.connect() as conn:
            conn.execute("DELETE FROM schema_migrations WHERE version = 9999")
    assert _versions(store) == [v for v, _, _ in MIGRATIONS]
