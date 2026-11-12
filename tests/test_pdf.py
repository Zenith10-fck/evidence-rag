import io

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.ingest_pdf import PdfRejected, check_pdf_upload, extract_pages
from tests.conftest import ROOT


def _pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        if text:
            c.drawString(72, 800, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_extracts_text_per_page_with_page_numbers():
    pages = extract_pages(_pdf(["page one text", "page two text"]), max_pages=10)
    assert [n for n, _ in pages] == [1, 2]
    assert "page two" in pages[1][1]


def test_corpus_pdf_has_three_pages():
    data = (ROOT / "data" / "corpus" / "warranty.pdf").read_bytes()
    assert len(extract_pages(data, 50)) == 3


def test_rejects_non_pdf_signature():
    with pytest.raises(PdfRejected, match="not a PDF"):
        check_pdf_upload(b"hello world", "application/pdf", 10)


def test_rejects_wrong_content_type():
    with pytest.raises(PdfRejected, match="content type"):
        check_pdf_upload(_pdf(["x"]), "text/plain", 10)


def test_rejects_too_large():
    with pytest.raises(PdfRejected, match="larger than"):
        check_pdf_upload(b"%PDF-" + b"0" * (1024 * 1024 + 10), "application/pdf", 1)


def test_rejects_too_many_pages():
    with pytest.raises(PdfRejected, match="limit is 2"):
        extract_pages(_pdf(["a", "b", "c"]), max_pages=2)


def test_rejects_pdf_without_text_like_a_scan():
    with pytest.raises(PdfRejected, match="no extractable text"):
        extract_pages(_pdf(["", ""]), max_pages=10)


def test_rejects_broken_pdf():
    with pytest.raises(PdfRejected):
        extract_pages(b"%PDF-1.4 garbage", max_pages=10)
