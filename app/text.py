"""Whitespace normalisation and quote location.

The evidence check is deliberately strict: the only thing normalised is whitespace (runs of spaces,
tabs and line breaks become one space). Case, punctuation, hyphenation and ligatures are NOT
normalised, so a quote that differs in any of those is rejected.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise whitespace and keep, for each output character, its index in `text`."""
    out: list[str] = []
    index_map: list[int] = []
    in_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            in_ws = True
            continue
        if in_ws and out:
            out.append(" ")
            index_map.append(i - 1)
        in_ws = False
        out.append(ch)
        index_map.append(i)
    return "".join(out), index_map


def locate_quote(chunk_text: str, quote: str) -> tuple[int, int] | None:
    """Return (start, end) of `quote` inside `chunk_text` (original indices), or None.

    Matching is exact after whitespace normalisation of both sides. A match must not start or end
    inside a word or number: "30 days" does not match inside "130 days", "ustomers" does not
    match inside "Customers". Otherwise a partial number could be quoted as evidence.
    """
    needle = normalize_ws(quote)
    if not needle:
        return None
    haystack, index_map = _normalized_with_map(chunk_text)
    pos = haystack.find(needle)
    while pos >= 0:
        end = pos + len(needle)
        if _clean_start(haystack, pos, needle) and _clean_end(haystack, end, needle):
            return index_map[pos], index_map[end - 1] + 1
        pos = haystack.find(needle, pos + 1)
    return None


def _clean_start(h: str, pos: int, needle: str) -> bool:
    """The match does not begin inside a word, or inside a number like '1,500' or '8.40'."""
    if pos == 0 or not needle[0].isalnum():
        return True
    before = h[pos - 1]
    if before.isalnum():
        return False
    return not (needle[0].isdigit() and before in ",." and pos >= 2 and h[pos - 2].isdigit())


def _clean_end(h: str, end: int, needle: str) -> bool:
    """The match does not end inside a word, or inside a number like '1,500' or '8.40'."""
    if end == len(h) or not needle[-1].isalnum():
        return True
    after = h[end]
    if after.isalnum():
        return False
    return not (
        needle[-1].isdigit() and after in ",." and end + 1 < len(h) and h[end + 1].isdigit()
    )


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text: str) -> list[str]:
    """Numbers as written (e.g. '30', '8.40', '1,500'), used by the evidence check."""
    return _NUMBER.findall(text)


def canonical_number(num: str) -> str:
    """'1,500' -> '1500'; '8.40' stays '8.40'."""
    return num.replace(",", "")
