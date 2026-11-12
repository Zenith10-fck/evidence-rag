"""Split document pages into chunks.

Rules:
- A chunk never crosses a page boundary, so each chunk (and each citation) has exactly one page.
- Units are sentences, list lines and headings. A heading is glued to the unit after it.
- Units are packed until `max_chars`; the next chunk starts with the last sentence of the previous
  one (one-sentence overlap), so a two-sentence quote is always whole inside some chunk.
- A single unit longer than `max_chars` becomes its own chunk (never cut mid-sentence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    position: int
    page: int | None
    text: str


@dataclass(frozen=True)
class _Unit:
    text: str
    is_heading: bool = False
    starts_paragraph: bool = False


def split_sentences(paragraph: str) -> list[str]:
    flat = " ".join(paragraph.split())
    if not flat:
        return []
    return [s for s in _SENTENCE_END.split(flat) if s]


def _flush_prose(buffer: list[str], units: list[_Unit], first_in_para: bool) -> bool:
    """Turn buffered prose lines into sentence units. Returns the new `first_in_para`."""
    if not buffer:
        return first_in_para
    for j, sentence in enumerate(split_sentences(" ".join(buffer))):
        units.append(_Unit(sentence, starts_paragraph=first_in_para and j == 0))
    buffer.clear()
    return False


def _units(page_text: str) -> list[_Unit]:
    units: list[_Unit] = []
    for para in _PARAGRAPH_SPLIT.split(page_text.strip()):
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        buffer: list[str] = []
        first_in_para = True
        for line in lines:
            if line.startswith("#") or line.startswith("- "):
                first_in_para = _flush_prose(buffer, units, first_in_para)
                units.append(_Unit(line, is_heading=line.startswith("#"), starts_paragraph=True))
                first_in_para = False
            else:
                buffer.append(line)
        _flush_prose(buffer, units, first_in_para)
    return units


def _join(units: list[_Unit]) -> str:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i > 0:
            parts.append("\n" if (unit.starts_paragraph or units[i - 1].is_heading) else " ")
        parts.append(unit.text)
    return "".join(parts)


def chunk_page(page_text: str, max_chars: int = 700) -> list[str]:
    units = _units(page_text)
    chunks: list[list[_Unit]] = []
    current: list[_Unit] = []

    def size(us: list[_Unit]) -> int:
        return sum(len(u.text) + 1 for u in us)

    for unit in units:
        if current and size(current) + len(unit.text) > max_chars and not current[-1].is_heading:
            chunks.append(current)
            last = current[-1]
            # one-sentence overlap (never repeat a heading or list line on its own)
            current = [last] if not last.is_heading else []
        current.append(unit)
    if current and (chunks == [] or not all(u.is_heading for u in current)):
        chunks.append(current)  # a heading-only page still becomes a chunk
    elif current:
        chunks[-1].extend(current)  # trailing headings join the last chunk
    return [_join(c) for c in chunks]


def chunk_pages(pages: list[tuple[int | None, str]], max_chars: int = 700) -> list[Chunk]:
    """`pages` is a list of (page_number_or_None, text). Positions are global across pages."""
    out: list[Chunk] = []
    for page_no, text in pages:
        for piece in chunk_page(text, max_chars=max_chars):
            out.append(Chunk(position=len(out), page=page_no, text=piece))
    return out
