"""HTTP layer: guards, status codes, the /ask flow with fakes, demo mode, config fail-early."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.config import ConfigError, Settings
from app.corpus import chunk_corpus, load_corpus
from app.guards import key_matches
from app.main import Services, create_app
from app.schemas import NOT_IN_DOCUMENTS
from tests.conftest import ROOT, FakeLLM, make_settings

H = {"X-API-Key": "test-key"}


@pytest.fixture
def live_client(store, fake_embedder, collection):
    llm = FakeLLM([])
    settings = make_settings()
    services = Services(settings=settings, store=store, embedder=fake_embedder, llm=llm)
    with TestClient(create_app(settings, services)) as client:
        client.llm = llm  # type: ignore[attr-defined]
        client.collection = collection  # type: ignore[attr-defined]
        yield client


# ---------- config and guards (no database) ----------


def test_config_fails_early_with_a_clear_message():
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        Settings(_env_file=None, evidence_rag_mode="live", database_url="", api_key="").check()
    Settings(_env_file=None, evidence_rag_mode="demo").check()  # demo needs nothing


def test_key_comparison():
    assert key_matches("abc", "abc")
    assert not key_matches("abd", "abc")
    assert not key_matches(None, "abc")
    assert not key_matches("abc", "")  # an empty server key never matches


@pytest.fixture
def demo_client(tmp_path):
    """Demo mode with a recordings file built for the test (a fixture, not a measured result)."""
    docs = load_corpus(ROOT / "data" / "corpus")
    chunks = chunk_corpus(docs, "harbor-goods")
    target = next(
        c for c in chunks.values() if "within 30 days of delivery" in c.text and c.doc_id == "returns-policy"
    )
    quote = next(s for s in target.text.split(". ") if "30 days" in s)
    if not quote.endswith("."):
        quote += "."
    good = {"answer": "30 days.", "abstained": False,
            "citations": [{"doc_id": target.doc_id, "chunk_id": target.chunk_id, "quote": quote}]}
    bad = {"answer": "90 days.", "abstained": False,
           "citations": [{"doc_id": target.doc_id, "chunk_id": target.chunk_id, "quote": "Ninety days for everything."}]}
    recordings = {
        "source": "test fixture",
        "recordings": [
            {"question": "Return window?", "mode": "hybrid", "question_id": "T1",
             "retrieved_chunk_ids": [target.chunk_id],
             "attempts": [{"raw_text": json.dumps(good), "stop_reason": "end_turn"}]},
            {"question": "Tampered?", "mode": "hybrid", "question_id": "T2",
             "retrieved_chunk_ids": [target.chunk_id],
             "attempts": [{"raw_text": json.dumps(bad), "stop_reason": "end_turn"},
                          {"raw_text": json.dumps(bad), "stop_reason": "end_turn"}]},
        ],
    }
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(recordings))
    settings = make_settings(evidence_rag_mode="demo", recordings_path=str(path), api_key="")
    with TestClient(create_app(settings)) as client:
        yield client


def test_health_and_ui_and_docs_in_demo(demo_client):
    assert demo_client.get("/health").json() == {"status": "ok", "mode": "demo"}
    assert "evidence-rag" in demo_client.get("/ui").text
    assert demo_client.get("/docs").status_code == 200
    assert demo_client.get("/openapi.json").json()["info"]["title"] == "evidence-rag"


def test_demo_replays_through_the_evidence_check(demo_client):
    r = demo_client.post("/ask", json={"question": "Return window?"}).json()
    assert r["status"] == "answered" and r["run_mode"] == "demo"
    assert r["citations"][0]["doc_id"] == "returns-policy"
    tampered = demo_client.post("/ask", json={"question": "Tampered?"}).json()
    assert tampered["status"] == "needs_human" and tampered["answer"] is None


def test_demo_unrecorded_question_is_not_made_up(demo_client):
    r = demo_client.post("/ask", json={"question": "Something new?"})
    assert r.status_code == 404 and "not recorded" in r.json()["detail"]


def test_demo_lists_documents_and_is_read_only(demo_client):
    docs = demo_client.get("/documents").json()["documents"]
    assert len(docs) == 20
    assert {d["pages"] for d in docs if d["source_type"] == "pdf"} == {3}
    r = demo_client.post("/documents", json={"doc_id": "x", "title": "x", "text": "x"})
    assert r.status_code == 409


def test_demo_questions_endpoint(demo_client):
    qs = demo_client.get("/demo/questions").json()
    assert qs["mode"] == "demo" and {q["id"] for q in qs["questions"]} == {"T1", "T2"}


# ---------- live mode with Postgres and fakes ----------

pytest_db = pytest.mark.db


@pytest_db
def test_missing_or_wrong_key_is_401(live_client):
    assert live_client.get("/documents").status_code == 401
    assert live_client.get("/documents", headers={"X-API-Key": "nope"}).status_code == 401
    assert live_client.get("/health").status_code == 200  # public


@pytest_db
def test_question_length_and_mode_are_validated(live_client):
    c = live_client.collection
    assert live_client.post("/ask", headers=H, json={"question": "x" * 501, "collection_id": c}).status_code == 422
    assert live_client.post("/ask", headers=H, json={"question": "   ", "collection_id": c}).status_code == 422
    assert live_client.post("/ask", headers=H, json={"question": "ok?", "mode": "magic"}).status_code == 422
    r = live_client.post("/ask", headers=H, json={"question": "ok?", "k": 50, "collection_id": c})
    assert r.status_code == 409  # k is ignored (not a parameter); empty collection -> 409


@pytest_db
def test_bad_ids_are_422(live_client):
    r = live_client.post("/documents", headers=H, json={"doc_id": "../etc", "title": "t", "text": "x"})
    assert r.status_code == 422


@pytest_db
def test_full_flow_text_document_then_ask(live_client):
    c = live_client.collection
    body = {"collection_id": c, "doc_id": "returns-policy", "title": "Returns Policy", "version": "2026-01",
            "text": "Customers may return unused items within 30 days of delivery."}
    r = live_client.post("/documents", headers=H, json=body)
    assert r.status_code == 200 and r.json()["status"] == "created"
    assert live_client.post("/documents", headers=H, json=body).json()["status"] == "unchanged"
    live_client.llm.responses += [
        {"answer": "30 days.", "abstained": False,
         "citations": [{"doc_id": "returns-policy", "chunk_id": f"{c}:returns-policy:0",
                        "quote": "Customers may return unused items within 30 days of delivery."}]},
        {"answer": NOT_IN_DOCUMENTS, "abstained": True, "citations": []},
    ]
    r = live_client.post("/ask", headers=H, json={"question": "How long to return?", "collection_id": c}).json()
    assert r["status"] == "answered" and r["citations"][0]["page"] is None
    assert r["citations"][0]["chunk_text"][r["citations"][0]["quote_start"]:r["citations"][0]["quote_end"]].startswith("Customers")
    r = live_client.post("/ask", headers=H, json={"question": "Antarctica?", "collection_id": c}).json()
    assert r["status"] == "abstained" and r["answer"] == NOT_IN_DOCUMENTS
    listed = live_client.get(f"/documents?collection_id={c}", headers=H).json()["documents"]
    assert listed[0]["chunks"] == 1


@pytest_db
def test_retrieve_route(live_client):
    c = live_client.collection
    live_client.post("/documents", headers=H, json={"collection_id": c, "doc_id": "d", "title": "T", "text": "Alpha beta gamma."})
    r = live_client.post("/retrieve", headers=H, json={"question": "beta", "collection_id": c}).json()
    assert r["chunks"][0]["doc_id"] == "d"


def _pdf_bytes(pages):
    buf = io.BytesIO()
    cv = canvas.Canvas(buf)
    for t in pages:
        cv.drawString(72, 750, t)
        cv.showPage()
    cv.save()
    return buf.getvalue()


@pytest_db
def test_pdf_upload(live_client):
    c = live_client.collection
    files = {"file": ("terms.pdf", _pdf_bytes(["Net 30 payment terms.", "Late fee is 2 percent."]), "application/pdf")}
    r = live_client.post("/documents/pdf", headers=H, files=files,
                         data={"doc_id": "terms", "title": "Terms", "collection_id": c})
    assert r.status_code == 200, r.text
    assert r.json()["pages"] == 2
    bad = {"file": ("x.pdf", b"not a pdf", "application/pdf")}
    r = live_client.post("/documents/pdf", headers=H, files=bad, data={"doc_id": "x", "title": "X", "collection_id": c})
    assert r.status_code == 422 and "not a PDF" in r.json()["detail"]
    blank = {"file": ("b.pdf", _pdf_bytes(["", ""]), "application/pdf")}
    r = live_client.post("/documents/pdf", headers=H, files=blank, data={"doc_id": "b", "title": "B", "collection_id": c})
    assert r.status_code == 422 and "no extractable text" in r.json()["detail"]


@pytest_db
def test_ask_without_anthropic_key_is_503(store, fake_embedder, collection):
    settings = make_settings()
    services = Services(settings=settings, store=store, embedder=fake_embedder)
    with TestClient(create_app(settings, services)) as client:
        client.post("/documents", headers=H, json={"collection_id": collection, "doc_id": "d", "title": "T", "text": "Alpha."})
        r = client.post("/ask", headers=H, json={"question": "alpha?", "collection_id": collection})
        assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_demo_rejects_another_collection(demo_client):
    r = demo_client.post("/ask", json={"question": "Return window?", "collection_id": "someone-else"})
    assert r.status_code == 404 and "recorded collection" in r.json()["detail"]
    assert demo_client.post("/ask", json={"question": "Return window?", "collection_id": "harbor-goods"}).status_code == 200


def test_metrics_and_samples_endpoints(demo_client):
    m = demo_client.get("/metrics").json()
    assert "retrieval_eval" in m or m == {"note": "not measured yet"}
    # demo mode only offers questions that were recorded; the fixture records none from the set
    assert demo_client.get("/samples").json() == {"groups": []}


def test_samples_in_live_mode_group_the_test_set():
    from app.main import Services

    settings = make_settings()
    with TestClient(create_app(settings, Services(settings=settings))) as client:
        groups = client.get("/samples").json()["groups"]
    names = [g["group"] for g in groups]
    assert "Decoys" in names and "Not in the documents" in names
    assert all(1 <= len(g["items"]) <= 5 for g in groups)
    assert all(set(i) == {"id", "question"} for g in groups for i in g["items"])  # no answer keys leak
