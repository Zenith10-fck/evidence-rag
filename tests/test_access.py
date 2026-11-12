"""Caller keys, usage limits, request logs and CORS at the HTTP layer.

None of these need the database or any API key: live-mode apps are built with `Services` that has
no store, so a request that passes auth and limits ends in a 503 "database is not configured".
That 503 is the proof it got past the guards (a 401 or 429 would have come first).
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigError, Settings
from app.guards import caller_for_key
from app.logs import LOGGER_NAME, JsonFormatter, log_event, setup_logging
from app.main import Services, create_app
from tests.conftest import make_settings

ACME = "acme-key-0123456789"
BETA = "beta-key-0123456789"
TWO_CALLERS = f"acme:{ACME},beta:{BETA}"
ASK = {"question": "ok?"}


@pytest.fixture(autouse=True)
def _clean_logger():
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for h in ours(logger):
        logger.removeHandler(h)


def ours(logger: logging.Logger) -> list[logging.Handler]:
    """Only the handler setup_logging adds (pytest attaches its own capture handlers too)."""
    return [h for h in logger.handlers if getattr(h, "_evidence_rag_handler", False)]


def live_client(**overrides) -> TestClient:
    settings = make_settings(**overrides)
    return TestClient(create_app(settings, Services(settings=settings)))


def demo_client(**overrides) -> TestClient:
    settings = make_settings(evidence_rag_mode="demo", api_key="", **overrides)
    return TestClient(create_app(settings))


# ---------- caller keys ----------


def test_callers_come_from_api_key_and_api_keys():
    assert Settings(_env_file=None, api_key="").callers == {}
    assert Settings(_env_file=None, api_key="x" * 8).callers == {"default": "x" * 8}
    s = Settings(_env_file=None, api_key="x" * 8, api_keys=TWO_CALLERS)
    assert s.callers == {"default": "x" * 8, "acme": ACME, "beta": BETA}


@pytest.mark.parametrize(
    "bad, message",
    [
        ("oops", "name:key"),
        ("acme:", "name:key"),
        (":" + ACME, "name:key"),
        (f"acme:{ACME},acme:{BETA}", "used twice"),
        ("acme:short", "shorter than 16"),
        (f"acme:{ACME},beta:{ACME}", "same key"),
        (f"bad name:{ACME}", "letters, digits"),
    ],
)
def test_malformed_api_keys_fail_instead_of_guessing(bad, message):
    with pytest.raises(ConfigError, match=message):
        Settings(_env_file=None, api_keys=bad).callers  # noqa: B018


def test_live_mode_needs_at_least_one_key():
    s = Settings(_env_file=None, evidence_rag_mode="live", database_url="postgresql://x", api_key="")
    with pytest.raises(ConfigError, match="API_KEY"):
        s.check()
    Settings(
        _env_file=None, evidence_rag_mode="live", database_url="postgresql://x", api_keys=TWO_CALLERS
    ).check()


def test_caller_for_key():
    callers = {"acme": ACME, "beta": BETA}
    assert caller_for_key(BETA, callers) == "beta"
    assert caller_for_key("wrong", callers) is None
    assert caller_for_key(None, callers) is None
    assert caller_for_key("", callers) is None
    assert caller_for_key(ACME, {}) is None


def test_startup_stops_on_a_malformed_api_keys_setting():
    with pytest.raises(ConfigError, match="name:key"):
        with live_client(api_keys="oops"):
            pass


def test_each_caller_key_works_and_a_wrong_key_does_not():
    with live_client(api_key="", api_keys=TWO_CALLERS) as c:
        assert c.get("/documents").status_code == 401
        assert c.get("/documents", headers={"X-API-Key": "nope"}).status_code == 401
        for key in (ACME, BETA):
            r = c.get("/documents", headers={"X-API-Key": key})
            assert r.status_code == 503 and "database is not configured" in r.json()["detail"]
        assert c.get("/health").status_code == 200  # public


# ---------- usage limits ----------


def test_per_minute_limit_is_per_caller_and_returns_retry_after():
    with live_client(api_key="", api_keys=TWO_CALLERS, ask_per_minute=2, ask_per_day=0) as c:
        acme = {"X-API-Key": ACME}
        beta = {"X-API-Key": BETA}
        assert [c.post("/retrieve", headers=acme, json=ASK).status_code for _ in range(2)] == [503, 503]
        r = c.post("/retrieve", headers=acme, json=ASK)
        assert r.status_code == 429
        assert 1 <= int(r.headers["Retry-After"]) <= 60
        assert "per-minute" in r.json()["detail"]
        assert c.post("/ask", headers=acme, json=ASK).status_code == 429  # /ask shares the quota
        assert c.post("/retrieve", headers=beta, json=ASK).status_code == 503  # beta is untouched


def test_daily_limit():
    with live_client(api_key="test-key", ask_per_minute=0, ask_per_day=1) as c:
        h = {"X-API-Key": "test-key"}
        assert c.post("/retrieve", headers=h, json=ASK).status_code == 503
        r = c.post("/retrieve", headers=h, json=ASK)
        assert r.status_code == 429 and "daily" in r.json()["detail"]
        assert int(r.headers["Retry-After"]) >= 1


def test_wrong_keys_do_not_use_up_anyones_quota():
    with live_client(api_key="test-key", ask_per_minute=1, ask_per_day=0) as c:
        for _ in range(5):
            assert c.post("/retrieve", headers={"X-API-Key": "nope"}, json=ASK).status_code == 401
        assert c.post("/retrieve", headers={"X-API-Key": "test-key"}, json=ASK).status_code == 503


def test_routes_that_cost_nothing_are_not_limited():
    with live_client(api_key="test-key", ask_per_minute=1, ask_per_day=1) as c:
        h = {"X-API-Key": "test-key"}
        for _ in range(5):
            assert c.get("/documents", headers=h).status_code == 503  # not 429
            assert c.get("/health").status_code == 200


def test_demo_mode_limits_per_client_ip_and_reports_remaining():
    with demo_client(demo_per_minute=2) as c:
        recorded = c.get("/demo/questions").json()["questions"][0]
        body = {"question": recorded["question"], "mode": recorded["modes"][0]}
        first = c.post("/ask", json=body)
        assert first.status_code == 200
        assert first.headers["X-RateLimit-Remaining-Minute"] == "1"
        assert c.post("/ask", json=body).status_code == 200
        r = c.post("/ask", json=body)
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
        assert c.get("/health").status_code == 200


# ---------- logs ----------


def log_lines(capsys) -> list[dict]:
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def test_request_log_has_ids_and_counts_but_never_text_or_keys(capsys):
    secret_question = "SECRET-QUESTION-TEXT-XYZ about our suppliers"
    with live_client(api_key="", api_keys=TWO_CALLERS) as c:
        r = c.post(
            "/retrieve", headers={"X-API-Key": ACME}, json={"question": secret_question}
        )
        c.get("/documents", headers={"X-API-Key": "wrong-key-that-must-not-appear"})
        c.get("/some/secret/path/probe")
    out = capsys.readouterr()
    assert secret_question not in out.out and "SECRET-QUESTION" not in out.out
    assert ACME not in out.out and BETA not in out.out
    assert "wrong-key-that-must-not-appear" not in out.out
    assert "secret/path/probe" not in out.out  # unmatched routes are logged as "unmatched"

    lines = [json.loads(x) for x in out.out.splitlines() if x.startswith("{")]
    started = next(x for x in lines if x["event"] == "started")
    assert started["callers"] == ["acme", "beta"]
    requests = [x for x in lines if x["event"] == "request"]
    retrieve = next(x for x in requests if x["route"] == "/retrieve")
    assert retrieve["caller"] == "acme" and retrieve["status"] == 503
    assert retrieve["question_len"] == len(secret_question)
    assert retrieve["request_id"] == r.headers["X-Request-ID"]
    assert isinstance(retrieve["duration_ms"], int)
    unauth = next(x for x in requests if x["route"] == "/documents")
    assert unauth["status"] == 401 and unauth["caller"] is None
    assert next(x for x in requests if x["route"] == "unmatched")["status"] == 404


def test_ask_log_has_outcome_and_token_counts_in_demo(capsys):
    with demo_client() as c:
        recorded = c.get("/demo/questions").json()["questions"][0]
        r = c.post("/ask", json={"question": recorded["question"], "mode": recorded["modes"][0]})
        assert r.status_code == 200
        status = r.json()["status"]
    ask = next(x for x in log_lines(capsys) if x.get("route") == "/ask")
    assert ask["caller"] == "demo" and ask["status"] == 200
    assert ask["answer_status"] == status and ask["run_mode"] == "demo"
    assert {"retried", "citations", "input_tokens", "output_tokens", "question_len"} <= set(ask)
    assert recorded["question"] not in json.dumps(ask)


def test_every_response_carries_a_request_id():
    with live_client() as c:
        ids = {c.get("/health").headers["X-Request-ID"], c.get("/health").headers["X-Request-ID"]}
        assert len(ids) == 2 and all(len(i) == 12 for i in ids)
        assert "X-Request-ID" in c.get("/documents").headers  # also on a 401


def test_setup_logging_is_idempotent_and_formats(capsys):
    setup_logging("json", "INFO")
    logger = setup_logging("json", "INFO")
    assert len(ours(logger)) == 1
    log_event(logger, "hello", n=3)
    record = json.loads(capsys.readouterr().out.strip())
    assert record["event"] == "hello" and record["n"] == 3 and record["level"] == "info"

    logger = setup_logging("text", "INFO")
    log_event(logger, "hello", n=3)
    assert "hello n=3" in capsys.readouterr().out

    rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "boom", None, None)
    assert json.loads(JsonFormatter().format(rec))["level"] == "error"


# ---------- CORS ----------


def preflight(c: TestClient, origin: str):
    return c.options(
        "/ask",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key,content-type",
        },
    )


def test_cors_allows_only_the_listed_origin():
    with live_client(cors_origins="https://app.example.com, https://b.example.com") as c:
        ok = preflight(c, "https://app.example.com")
        assert ok.status_code == 200
        assert ok.headers["access-control-allow-origin"] == "https://app.example.com"
        other = preflight(c, "https://evil.example.com")
        assert "access-control-allow-origin" not in other.headers
        assert other.status_code == 400


def test_cors_is_off_by_default():
    with live_client() as c:
        r = preflight(c, "https://app.example.com")
        assert "access-control-allow-origin" not in r.headers
