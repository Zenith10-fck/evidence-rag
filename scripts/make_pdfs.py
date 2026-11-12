"""Render the four PDF documents from their `<doc_id>.source.md` files.

- One source section (split by `<!-- pagebreak -->`) = one PDF page, so the page numbers in the
  answer keys are real. The script fails loudly if a section overflows onto a second page.
- Standard font (Helvetica), text only, so text extraction is reliable. No hyphenation.
- Repeatable: reportlab's invariant mode removes creation dates and random ids, so the same input
  gives the same bytes.

Usage: uv run python scripts/make_pdfs.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab import rl_config

rl_config.invariant = 1  # must be set before documents are built

from pypdf import PdfReader  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.corpus import PAGEBREAK, parse_front_matter  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
PDF_DOCS = ["supplier-terms-alpine", "supplier-terms-trailworks", "warranty", "data-retention"]

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=11, leading=14.5, spaceAfter=7)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=10)
H2 = ParagraphStyle(
    "h2", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=6, spaceAfter=6
)
SMALL = ParagraphStyle("small", fontName="Helvetica", fontSize=9, leading=12, spaceAfter=10)


def flowables_for(section: str) -> list:
    out: list = []
    for block in section.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("## "):
            out.append(Paragraph(escape(block[3:].strip()), H2))
        elif block.startswith("# "):
            out.append(Paragraph(escape(block[2:].strip()), H1))
        else:
            out.append(Paragraph(escape(" ".join(block.split())), BODY))
    return out


def render(doc_id: str) -> Path:
    source = CORPUS / f"{doc_id}.source.md"
    meta, body = parse_front_matter(source.read_text("utf-8"))
    sections = [s for s in body.split(PAGEBREAK) if s.strip()]
    story: list = []
    for i, section in enumerate(sections):
        if i == 0:
            story.append(
                Paragraph(
                    escape(
                        f"Harbor Goods internal document. Version {meta.get('version', '')}, "
                        f"effective {meta.get('effective', '')}. Synthetic data."
                    ),
                    SMALL,
                )
            )
        story.extend(flowables_for(section))
        if i < len(sections) - 1:
            story.append(PageBreak())
    story.append(Spacer(1, 1))
    out = CORPUS / f"{doc_id}.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=meta.get("title", doc_id),
        author="Harbor Goods (synthetic)",
        subject="Synthetic reference corpus",
        creator="evidence-rag make_pdfs.py",
    )
    doc.build(story)
    pages = len(PdfReader(str(out)).pages)
    if pages != len(sections):
        raise SystemExit(
            f"{doc_id}: rendered {pages} pages for {len(sections)} sections; "
            "a section overflowed. Shorten it in make_corpus.py."
        )
    return out


def main() -> None:
    for doc_id in PDF_DOCS:
        out = render(doc_id)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
        print(f"{out.relative_to(ROOT)}  pages={len(PdfReader(str(out)).pages)}  sha256={digest}")


if __name__ == "__main__":
    main()
