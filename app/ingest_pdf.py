"""PDF to text, one string per page.

Library: pypdf. Chosen because it is pure Python (no system packages, works the same on Windows,
Linux and in a slim Docker image), is actively maintained, and its per-page `extract_text()` is
enough for text-based PDFs. Scanned PDFs (images of text) are out of scope and rejected.

Known limits (not fixed silently): extraction may change line breaks (handled by whitespace
normalisation in the evidence check); hyphenated words split across lines and ligature glyphs would
NOT match a quote typed without them.
"""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

PDF_MAGIC = b"%PDF-"


class PdfRejected(ValueError):
    """The upload is not an acceptable PDF. The message is safe to show to the caller."""


def check_pdf_upload(data: bytes, content_type: str | None, max_mb: int) -> None:
    if content_type not in (None, "", "application/pdf", "application/octet-stream"):
        raise PdfRejected(f"content type must be application/pdf, got {content_type}")
    if len(data) == 0:
        raise PdfRejected("file is empty")
    if len(data) > max_mb * 1024 * 1024:
        raise PdfRejected(f"file is larger than {max_mb} MB")
    if not data.startswith(PDF_MAGIC):
        raise PdfRejected("file is not a PDF (missing %PDF- signature)")


def extract_pages(data: bytes, max_pages: int) -> list[tuple[int, str]]:
    """Return [(page_number_1_based, text), ...]. Raises PdfRejected."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfRejected("encrypted PDFs are not supported")
        n = len(reader.pages)
    except PdfRejected:
        raise
    except PdfReadError as exc:
        raise PdfRejected(f"could not read PDF: {exc}") from exc
    except Exception as exc:  # malformed PDFs raise many types inside pypdf
        raise PdfRejected(f"could not read PDF ({type(exc).__name__})") from exc
    if n == 0:
        raise PdfRejected("PDF has no pages")
    if n > max_pages:
        raise PdfRejected(f"PDF has {n} pages; the limit is {max_pages}")
    try:
        pages = [(i + 1, (page.extract_text() or "")) for i, page in enumerate(reader.pages)]
    except Exception as exc:  # malformed PDFs raise many types inside pypdf
        raise PdfRejected(f"could not extract text from PDF ({type(exc).__name__})") from exc
    if not any(text.strip() for _, text in pages):
        raise PdfRejected("PDF has no extractable text (scanned PDFs are not supported)")
    return pages
