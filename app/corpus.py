"""Load the synthetic corpus from `data/corpus/`.

Markdown documents are `<doc_id>.md`. PDF documents are `<doc_id>.pdf`; their metadata (title,
version) is read from the generator's `<doc_id>.source.md`, and their text from the PDF itself,
page by page. Used by the ingest script, the measurement scripts and demo mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.chunking import Chunk, chunk_pages
from app.ingest_pdf import extract_pages
from app.store import make_chunk_id

PAGEBREAK = "<!-- pagebreak -->"


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    version: str
    source_type: str  # text | pdf
    pages: list[tuple[int | None, str]]
    raw_content: bytes


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split `---` front matter from the body. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    _, header, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in header.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body.lstrip("\n")


def load_corpus(corpus_dir: str | Path, max_pdf_pages: int = 50) -> list[CorpusDoc]:
    root = Path(corpus_dir)
    docs: list[CorpusDoc] = []
    for path in sorted(root.glob("*.md")):
        if path.name.endswith(".source.md"):
            continue
        raw = path.read_bytes()
        meta, body = parse_front_matter(raw.decode("utf-8"))
        docs.append(
            CorpusDoc(
                doc_id=meta.get("doc_id", path.stem),
                title=meta.get("title", path.stem),
                version=meta.get("version", ""),
                source_type="text",
                pages=[(None, body)],
                raw_content=raw,
            )
        )
    for pdf in sorted(root.glob("*.pdf")):
        source = pdf.with_name(pdf.stem + ".source.md")
        meta = parse_front_matter(source.read_text("utf-8"))[0] if source.exists() else {}
        raw = pdf.read_bytes()
        docs.append(
            CorpusDoc(
                doc_id=pdf.stem,
                title=meta.get("title", pdf.stem),
                version=meta.get("version", ""),
                source_type="pdf",
                pages=[(n, t) for n, t in extract_pages(raw, max_pdf_pages)],
                raw_content=raw,
            )
        )
    return sorted(docs, key=lambda d: d.doc_id)


@dataclass(frozen=True)
class MemoryChunk:
    chunk_id: str
    doc_id: str
    title: str
    position: int
    page: int | None
    text: str


def chunk_corpus(
    docs: list[CorpusDoc], collection_id: str, max_chars: int = 700
) -> dict[str, MemoryChunk]:
    """Chunk the corpus in memory with the same chunker and ids as the live ingest."""
    out: dict[str, MemoryChunk] = {}
    for doc in docs:
        chunks: list[Chunk] = chunk_pages(doc.pages, max_chars=max_chars)
        for c in chunks:
            cid = make_chunk_id(collection_id, doc.doc_id, c.position)
            out[cid] = MemoryChunk(cid, doc.doc_id, doc.title, c.position, c.page, c.text)
    return out


def load_questions(path: str | Path = "data/questions/questions.json") -> list[dict]:
    return json.loads(Path(path).read_text("utf-8"))
