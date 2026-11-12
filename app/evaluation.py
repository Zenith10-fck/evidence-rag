"""Metric definitions (see docs/MEASUREMENT.md). Pure functions over run records, unit-tested.

A run record is one question answered in one mode in one run:
{
  "question_id", "type", "tags", "mode", "run",
  "retrieved_doc_ids": [...], "retrieved_chunk_ids": [...],
  "status", "retried", "answer", "errors",
  "citations": [{"doc_id", "chunk_id", "page", "quote"}],     # shipped (checked) citations
  "attempts": [{"raw_text", "stop_reason", "parsed_ok", "abstained",
                "citations_total", "citations_valid", "evidence_ok"}],
  "final_evidence_ok": bool, "latency_s", "input_tokens", "output_tokens"
}
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median

from app.text import canonical_number, locate_quote, normalize_ws, numbers_in

ANSWERABLE = ("single", "multi")


# ---------- retrieval ----------


def recall_hit(expected_doc_ids: list[str], retrieved_doc_ids: list[str]) -> bool:
    """All expected documents are among the retrieved top-k chunks."""
    return set(expected_doc_ids) <= set(retrieved_doc_ids)


def reciprocal_rank(expected_doc_ids: list[str], retrieved_doc_ids: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return 1.0 / rank
    return 0.0


def retrieval_metrics(questions: list[dict], retrieved: dict[str, list[str]]) -> dict:
    """`retrieved` maps question id -> retrieved doc ids in rank order (one per chunk)."""
    rows = [q for q in questions if q["type"] in ANSWERABLE]

    def summarise(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "recall_at_k": None, "mrr": None, "hit_at_1": None}
        hits = [recall_hit(q["expected_doc_ids"], retrieved[q["id"]]) for q in subset]
        rr = [reciprocal_rank(q["expected_doc_ids"], retrieved[q["id"]]) for q in subset]
        return {
            "n": len(subset),
            "recall_at_k": round(sum(hits) / len(subset), 4),
            "mrr": round(sum(rr) / len(subset), 4),
            # stricter: the top-ranked chunk comes from an expected document
            "hit_at_1": round(sum(r == 1.0 for r in rr) / len(subset), 4),
        }

    out = summarise(rows)
    out["by_tag"] = {
        tag: summarise([q for q in rows if tag in q.get("tags", [])])
        for tag in ("exact-term", "paraphrase")
    }
    out["by_type"] = {t: summarise([q for q in rows if q["type"] == t]) for t in ANSWERABLE}
    out["misses"] = [
        q["id"] for q in rows if not recall_hit(q["expected_doc_ids"], retrieved[q["id"]])
    ]
    return out


# ---------- answers ----------

_WORD = re.compile(r"[a-z0-9]+")


def quote_matches_expected(quote: str, expected: str) -> bool:
    """A cited quote 'is' an expected quote if one contains the other (whitespace-normalised), or
    they share at least 80% of the shorter one's words AND carry exactly the same numbers (so
    "within 14 days" never matches "within 30 days")."""
    a, b = normalize_ws(quote).lower(), normalize_ws(expected).lower()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    if {canonical_number(n) for n in numbers_in(a)} != {canonical_number(n) for n in numbers_in(b)}:
        return False
    wa, wb = set(_WORD.findall(a)), set(_WORD.findall(b))
    shorter = min(len(wa), len(wb)) or 1
    return len(wa & wb) / shorter >= 0.8


def is_supported(record: dict, question: dict) -> bool:
    """Answered, every shipped citation valid, and at least one expected quote cited FROM AN
    EXPECTED DOCUMENT (a matching sentence from a superseded or decoy document does not count)."""
    if record["status"] != "answered" or not record["final_evidence_ok"]:
        return False
    return any(
        c["doc_id"] in question["expected_doc_ids"] and quote_matches_expected(c["quote"], e)
        for c in record["citations"]
        for e in question["expected_quotes"]
    )


def is_fully_supported(record: dict, question: dict) -> bool:
    """Stricter (added 2026-11-05 after the first run, see README): for EVERY expected document,
    the answer cites a quote from that document that matches one of the expected quotes.
    For single-document questions this equals `is_supported`; for two-document questions it
    catches answers that only cover one half."""
    if record["status"] != "answered" or not record["final_evidence_ok"]:
        return False
    return all(
        any(
            c["doc_id"] == doc and quote_matches_expected(c["quote"], e)
            for c in record["citations"]
            for e in question["expected_quotes"]
        )
        for doc in question["expected_doc_ids"]
    )


def verify_shipped(answer: str | None, citations: list[dict], chunk_texts: dict[str, str]) -> bool:
    """Re-check a shipped answer from its parts, separately from the status the pipeline set:
    at least one citation, every quote found (word-bounded) in the retrieved chunk it names, and
    every number in the answer inside a quote. Used for `unsupported_shipped`."""
    if not citations:
        return False
    quoted: set[str] = set()
    for c in citations:
        text = chunk_texts.get(c["chunk_id"])
        if text is None or locate_quote(text, c["quote"]) is None:
            return False
        quoted |= {canonical_number(n) for n in numbers_in(c["quote"])}
    return all(canonical_number(n) in quoted for n in numbers_in(answer or ""))


def page_correct(record: dict, question: dict) -> bool | None:
    """None if the question has no PDF page; else whether a cited page is an expected page."""
    expected = [p for p in question["expected_pages"] if p is not None]
    if not expected:
        return None
    return any(c["page"] in expected for c in record["citations"])


def contains_key_facts(record: dict, question: dict) -> bool:
    text = (record.get("answer") or "").lower()
    return all(s.lower() in text for s in question["must_contain"])


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[idx]


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def answer_metrics(records: list[dict], questions: dict[str, dict], price: dict) -> dict:
    """Aggregate metrics for one mode over all its runs."""
    if not records:
        return {}
    answerable = [r for r in records if r["type"] in ANSWERABLE]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    answered = [r for r in records if r["status"] == "answered"]

    cit_total = sum(a["citations_total"] for r in records for a in r["attempts"])
    cit_valid = sum(a["citations_valid"] for r in records for a in r["attempts"])

    page_checks = [
        page_correct(r, questions[r["question_id"]]) for r in answered if r["type"] in ANSWERABLE
    ]
    page_checks = [p for p in page_checks if p is not None]
    retried = [r for r in records if r["retried"]]
    latencies = [r["latency_s"] for r in records]
    in_tok = sum(r["input_tokens"] for r in records)
    out_tok = sum(r["output_tokens"] for r in records)
    cost = in_tok / 1e6 * price["input_per_mtok"] + out_tok / 1e6 * price["output_per_mtok"]

    def tag_block(tag: str) -> dict:
        sub = [r for r in answerable if tag in r["tags"]]
        return {
            "n": len(sub),
            "supported_answer_rate": _rate(
                sum(is_supported(r, questions[r["question_id"]]) for r in sub), len(sub)
            ),
            "recall_at_k": _rate(
                sum(
                    recall_hit(
                        questions[r["question_id"]]["expected_doc_ids"], r["retrieved_doc_ids"]
                    )
                    for r in sub
                ),
                len(sub),
            ),
        }

    return {
        "records": len(records),
        "recall_at_k": _rate(
            sum(
                recall_hit(questions[r["question_id"]]["expected_doc_ids"], r["retrieved_doc_ids"])
                for r in answerable
            ),
            len(answerable),
        ),
        "mrr": round(
            sum(
                reciprocal_rank(
                    questions[r["question_id"]]["expected_doc_ids"], r["retrieved_doc_ids"]
                )
                for r in answerable
            )
            / len(answerable),
            4,
        )
        if answerable
        else None,
        "valid_citation_rate": _rate(cit_valid, cit_total),
        "page_accuracy": _rate(sum(page_checks), len(page_checks)),
        "supported_answer_rate": _rate(
            sum(is_supported(r, questions[r["question_id"]]) for r in answerable), len(answerable)
        ),
        "fully_supported_answer_rate": _rate(
            sum(is_fully_supported(r, questions[r["question_id"]]) for r in answerable),
            len(answerable),
        ),
        "key_facts_in_answer_rate": _rate(
            sum(contains_key_facts(r, questions[r["question_id"]]) for r in answerable),
            len(answerable),
        ),
        "unsupported_shipped": sum(1 for r in answered if not r["final_evidence_ok"]),
        "abstention_recall": _rate(
            sum(r["status"] == "abstained" for r in unanswerable), len(unanswerable)
        ),
        "false_abstention": _rate(
            sum(r["status"] == "abstained" for r in answerable), len(answerable)
        ),
        "escalation_rate": _rate(sum(r["status"] == "needs_human" for r in records), len(records)),
        "retry_rate": _rate(len(retried), len(records)),
        "retry_fixed": _rate(sum(r["status"] != "needs_human" for r in retried), len(retried)),
        "latency_p50_s": round(median(latencies), 3),
        "latency_p95_s": round(percentile(latencies, 95), 3),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 4),
        "by_tag": {t: tag_block(t) for t in ("exact-term", "paraphrase")},
    }


def baseline_no_guard(records: list[dict]) -> dict:
    """What the guard buys: the FIRST attempt of each answer, as the model wrote it, before the
    evidence check and retry. `answers_with_invalid_quote` counts shipped (not abstained) answers
    with at least one quote that is not in ANY retrieved chunk (docs/MEASUREMENT.md). Answers
    with no citation at all are reported separately, not mixed in."""
    shipped = [r for r in records if r["attempts"] and r["attempts"][0]["parsed_ok"]]
    shipped = [r for r in shipped if not r["attempts"][0]["abstained"]]
    bad = [r for r in shipped if r["attempts"][0]["quotes_not_in_retrieved"] > 0]
    uncited = [r for r in shipped if r["attempts"][0]["citations_total"] == 0]
    # Any reason: invalid quote, wrong chunk, or a number in the answer that no quote contains.
    failed = [r for r in shipped if not r["attempts"][0]["evidence_ok"]]
    return {
        "answers_with_invalid_quote": len(bad),
        "of_answers": len(shipped),
        "rate": _rate(len(bad), len(shipped)),
        "answers_without_citations": len(uncited),
        "answers_failing_evidence_check": len(failed),
        "failing_rate": _rate(len(failed), len(shipped)),
    }


def describe_miss(record: dict, question: dict) -> str | None:
    """Why a record counts as a miss, in plain words; None if it is not a miss."""
    status = record["status"]
    if question["type"] == "unanswerable":
        if status == "answered":
            return "answered a question the documents do not answer (with valid quotes)"
        if status == "needs_human":
            return "escalated instead of abstaining: " + "; ".join(record["errors"][:2])
        return None
    if status == "abstained":
        if not recall_hit(question["expected_doc_ids"], record["retrieved_doc_ids"]):
            return "abstained; retrieval missed " + ", ".join(
                sorted(set(question["expected_doc_ids"]) - set(record["retrieved_doc_ids"]))
            )
        return "abstained although the answer was in the retrieved chunks"
    if status == "needs_human":
        return "needs_human after retry: " + "; ".join(record["errors"][:2])
    if not is_supported(record, question):
        if not recall_hit(question["expected_doc_ids"], record["retrieved_doc_ids"]):
            return "answered with valid quotes, but not the expected ones; retrieval missed " + (
                ", ".join(
                    sorted(set(question["expected_doc_ids"]) - set(record["retrieved_doc_ids"]))
                )
            )
        return "answered with valid quotes, but did not cite an expected quote"
    if not is_fully_supported(record, question):
        cited = {c["doc_id"] for c in record["citations"]}
        missing = sorted(set(question["expected_doc_ids"]) - cited)
        return "partial answer: no expected quote from " + ", ".join(missing or ["one document"])
    return None


def collect_misses(records: list[dict], questions: dict[str, dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = defaultdict(lambda: {"runs": [], "reasons": []})
    for r in records:
        reason = describe_miss(r, questions[r["question_id"]])
        if reason:
            g = grouped[(r["question_id"], r["mode"])]
            g["runs"].append(r["run"])
            if reason not in g["reasons"]:
                g["reasons"].append(reason)
    return [
        {
            "question_id": qid,
            "mode": mode,
            "runs_missed": sorted(g["runs"]),
            "what_went_wrong": " | ".join(g["reasons"]),
        }
        for (qid, mode), g in sorted(grouped.items())
    ]
