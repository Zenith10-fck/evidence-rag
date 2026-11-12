"""The test set itself: answer keys must be provable from the documents."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections import Counter

import pytest

from app.chunking import chunk_pages
from app.corpus import PAGEBREAK, load_corpus, parse_front_matter
from app.ingest_pdf import extract_pages
from app.text import normalize_ws
from tests.conftest import ROOT

CORPUS = ROOT / "data" / "corpus"
PDF_DOCS = {"supplier-terms-alpine", "supplier-terms-trailworks", "warranty", "data-retention"}


@pytest.fixture(scope="module")
def docs():
    return {d.doc_id: d for d in load_corpus(CORPUS)}


def test_twenty_documents_four_of_them_pdf(docs):
    assert len(docs) == 20
    assert {d for d, doc in docs.items() if doc.source_type == "pdf"} == PDF_DOCS
    for doc_id in PDF_DOCS:  # no duplicate markdown copy of a PDF document
        assert not (CORPUS / f"{doc_id}.md").exists()


def test_question_mix(questions):
    types = Counter(q["type"] for q in questions)
    assert types == {"single": 40, "multi": 10, "unanswerable": 10}
    tags = Counter(t for q in questions for t in q["tags"])
    assert tags["exact-term"] >= 8 and tags["paraphrase"] >= 8
    assert [q["id"] for q in questions] == [f"Q{i:03d}" for i in range(1, 61)]


def test_every_expected_quote_is_verbatim_in_its_source(docs, questions):
    for q in questions:
        assert len(q["expected_quotes"]) == len(q["expected_pages"]), q["id"]
        for quote, page in zip(q["expected_quotes"], q["expected_pages"], strict=True):
            needle = normalize_ws(quote)
            found_in = [
                (doc_id, p)
                for doc_id in q["expected_doc_ids"]
                for p, text in docs[doc_id].pages
                if needle in normalize_ws(text)
            ]
            assert found_in, f"{q['id']}: quote not found: {quote!r}"
            assert any(p == page for _, p in found_in), f"{q['id']}: wrong page {page}"


def test_pdf_pages_are_real(docs, questions):
    page2 = [
        q["id"] for q in questions if any(p is not None and p >= 2 for p in q["expected_pages"])
    ]
    assert len(page2) >= 4
    for doc_id in PDF_DOCS:
        source = (CORPUS / f"{doc_id}.source.md").read_text("utf-8")
        sections = [s for s in parse_front_matter(source)[1].split(PAGEBREAK) if s.strip()]
        pages = extract_pages((CORPUS / f"{doc_id}.pdf").read_bytes(), 50)
        assert len(pages) == len(sections) >= 2


def test_a_pdf_quote_crosses_a_line_break(docs, questions):
    """Proves whitespace normalisation matters: raw extracted text has a newline in the quote."""
    crossing = []
    for q in questions:
        for doc_id in q["expected_doc_ids"]:
            if doc_id not in PDF_DOCS:
                continue
            for quote in q["expected_quotes"]:
                for _, text in docs[doc_id].pages:
                    if quote not in text and normalize_ws(quote) in normalize_ws(text):
                        crossing.append(q["id"])
    assert crossing


def test_unanswerable_terms_appear_in_no_document(docs, questions):
    all_text = " ".join(t for d in docs.values() for _, t in d.pages).lower()
    for q in questions:
        if q["type"] != "unanswerable":
            continue
        assert not q["expected_quotes"] and not q["expected_doc_ids"]
        assert q["must_not_appear"], q["id"]
        for term in q["must_not_appear"]:
            assert term.lower() not in all_text, f"{q['id']}: {term!r} appears in a document"


def test_must_contain_is_inside_a_quote(questions):
    for q in questions:
        joined = " ".join(q["expected_quotes"])
        for s in q["must_contain"]:
            assert s in joined, f"{q['id']}: {s!r}"


def test_every_expected_quote_fits_inside_one_chunk(docs, questions):
    """The chunker must never split a quote the answer key needs."""
    for q in questions:
        for doc_id in q["expected_doc_ids"]:
            chunks = [normalize_ws(c.text) for c in chunk_pages(docs[doc_id].pages)]
            for quote in q["expected_quotes"]:
                if any(normalize_ws(quote) in normalize_ws(t) for _, t in docs[doc_id].pages):
                    assert any(normalize_ws(quote) in c for c in chunks), (q["id"], quote)


def _hashes() -> dict[str, str]:
    files = sorted(CORPUS.glob("*")) + [ROOT / "data" / "questions" / "questions.json"]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}


def test_generators_are_repeatable():
    before = _hashes()
    for script in ("make_corpus.py", "make_pdfs.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)], check=True, capture_output=True
        )
    assert _hashes() == before
