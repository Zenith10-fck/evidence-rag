"""The evidence check: the model's answer is untrusted until every citation is proven.

A citation is valid only if:
1. its `chunk_id` is one of the chunks retrieved for this question,
2. its `doc_id` matches that chunk,
3. its `quote` (whitespace-normalised) is an exact substring of that chunk's text.

An answer (not abstained) passes only if:
- it has at least one citation and EVERY citation is valid, and
- every number written in the answer appears in at least one valid cited quote.

"Every sentence is covered" is approximated by these two rules (documented limit): a sentence
without a number that states something the quotes do not support is not caught by code. The
measurement reports how often answers cite an expected quote, to show how well this works.

For valid citations the code (never the model) copies `title`, `page` and the quote position from
the stored chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.schemas import ModelAnswer, ModelCitation
from app.text import canonical_number, locate_quote, normalize_ws, numbers_in

MIN_QUOTE_CHARS = 8


class ChunkLike(Protocol):
    chunk_id: str
    doc_id: str
    title: str
    page: int | None
    text: str


@dataclass(frozen=True)
class CheckedCitation:
    doc_id: str
    title: str
    chunk_id: str
    page: int | None
    quote: str
    quote_start: int
    quote_end: int
    chunk_text: str


@dataclass
class EvidenceResult:
    ok: bool
    abstained: bool = False
    citations: list[CheckedCitation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_citation(
    i: int, cit: ModelCitation, by_id: dict[str, ChunkLike]
) -> tuple[CheckedCitation | None, str | None]:
    """Check one citation. Returns (checked, None) if valid, else (None, error message)."""
    chunk = by_id.get(cit.chunk_id)
    if chunk is None:
        return None, f"citation {i}: chunk_id {cit.chunk_id!r} is not one of the retrieved chunks"
    if cit.doc_id != chunk.doc_id:
        return None, (
            f"citation {i}: doc_id {cit.doc_id!r} does not match chunk {cit.chunk_id} "
            f"(which belongs to {chunk.doc_id!r})"
        )
    if len(normalize_ws(cit.quote)) < MIN_QUOTE_CHARS:
        return None, f"citation {i}: quote is too short to be evidence"
    span = locate_quote(chunk.text, cit.quote)
    if span is None:
        return None, (
            f"citation {i}: quote not found in chunk {cit.chunk_id}; "
            "copy the words exactly, do not paraphrase"
        )
    return (
        CheckedCitation(
            doc_id=chunk.doc_id,
            title=chunk.title,
            chunk_id=chunk.chunk_id,
            page=chunk.page,  # from the stored chunk, never from the model
            quote=chunk.text[span[0] : span[1]],
            quote_start=span[0],
            quote_end=span[1],
            chunk_text=chunk.text,
        ),
        None,
    )


def check_answer(answer: ModelAnswer, chunks: list[ChunkLike]) -> EvidenceResult:
    if answer.abstained:
        # A refusal needs no proof. Any citations it carries are ignored, never shown as evidence.
        return EvidenceResult(ok=True, abstained=True)

    by_id = {c.chunk_id: c for c in chunks}
    errors: list[str] = []
    valid: list[CheckedCitation] = []

    if not answer.answer.strip():
        errors.append("answer text is empty")
    if not answer.citations:
        errors.append("answer has no citations; cite an exact quote or abstain")

    for i, cit in enumerate(answer.citations, start=1):
        checked, error = validate_citation(i, cit, by_id)
        if error:
            errors.append(error)
        else:
            valid.append(checked)

    if valid:
        quoted_numbers = {canonical_number(n) for c in valid for n in numbers_in(c.quote)}
        for num in numbers_in(answer.answer):
            if canonical_number(num) not in quoted_numbers:
                errors.append(f"the number {num} in the answer does not appear in any cited quote")

    if errors:
        return EvidenceResult(ok=False, citations=valid, errors=errors)
    return EvidenceResult(ok=True, citations=valid)
