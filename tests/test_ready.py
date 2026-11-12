"""Readiness check and the catch-all error handler. No database or API key needed: the store is a
stand-in whose `ping` succeeds or fails on demand."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import Services, create_app
from app.store import StoreError
from tests.conftest import make_settings


class StubStore:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.pings = 0

    def ping(self) -> None:
        self.pings += 1
        if self.error:
            raise self.error


def _client(store, **kw) -> TestClient:
    settings = make_settings()
    app = create_app(settings, Services(settings=settings, store=store))
    return TestClient(app, **kw)


def test_ready_is_200_when_the_database_answers():
    store = StubStore()
    with _client(store) as client:
        r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "mode": "live", "database": "ok"}
    assert store.pings == 1


def test_ready_is_503_when_the_database_is_down_but_health_stays_200():
    store = StubStore(StoreError("cannot connect to Postgres: OperationalError"))
    with _client(store) as client:
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json() == {"detail": "database is not ready"}  # no internal details leak
        assert client.get("/health").status_code == 200  # alive, just not ready


def test_ready_and_health_need_no_api_key():
    with _client(StubStore()) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/health").status_code == 200


def test_unexpected_error_returns_only_the_request_id():
    settings = make_settings()
    app = create_app(settings, Services(settings=settings, store=StubStore()))

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret detail: sk-ant-not-a-real-key")

    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["detail"] == "internal error"
    assert body["request_id"]
    assert "secret" not in r.text and "sk-ant" not in r.text
