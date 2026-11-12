from app.evaluation import (
    answer_metrics,
    baseline_no_guard,
    collect_misses,
    percentile,
    quote_matches_expected,
    recall_hit,
    reciprocal_rank,
    retrieval_metrics,
)

Q = {
    "Q1": {
        "id": "Q1",
        "type": "single",
        "tags": ["exact-term"],
        "expected_doc_ids": ["a"],
        "expected_quotes": ["Net 30 days from invoice date."],
        "expected_pages": [2],
        "must_contain": ["30"],
    },
    "Q2": {
        "id": "Q2",
        "type": "multi",
        "tags": ["paraphrase"],
        "expected_doc_ids": ["a", "b"],
        "expected_quotes": ["x one", "y two"],
        "expected_pages": [None, None],
        "must_contain": [],
    },
    "Q3": {
        "id": "Q3",
        "type": "unanswerable",
        "tags": [],
        "expected_doc_ids": [],
        "expected_quotes": [],
        "expected_pages": [],
        "must_contain": [],
    },
}


def attempt(total, valid, abstained=False, ok=True):
    return {
        "raw_text": "",
        "stop_reason": "end_turn",
        "parsed_ok": True,
        "abstained": abstained,
        "citations_total": total,
        "citations_valid": valid,
        "evidence_ok": ok,
        "quotes_not_in_retrieved": total - valid,
    }


def rec(qid, status, citations=(), retrieved=("a",), attempts=None, retried=False, ok=True, run=1):
    return {
        "question_id": qid,
        "type": Q[qid]["type"],
        "tags": Q[qid]["tags"],
        "mode": "hybrid",
        "run": run,
        "retrieved_doc_ids": list(retrieved),
        "retrieved_chunk_ids": [],
        "status": status,
        "retried": retried,
        "answer": "30 days" if status == "answered" else None,
        "errors": ["quote not found"] if status == "needs_human" else [],
        "citations": list(citations),
        "attempts": attempts or [attempt(1, 1)],
        "final_evidence_ok": ok,
        "latency_s": 1.0,
        "input_tokens": 1000,
        "output_tokens": 100,
    }


def test_recall_and_mrr():
    assert recall_hit(["a", "b"], ["b", "c", "a"])
    assert not recall_hit(["a", "b"], ["b", "c"])
    assert reciprocal_rank(["a"], ["c", "a"]) == 0.5
    assert reciprocal_rank(["a"], ["c"]) == 0.0


def test_retrieval_metrics_ignores_unanswerable():
    m = retrieval_metrics(list(Q.values()), {"Q1": ["a"], "Q2": ["a", "c"]})
    assert m["n"] == 2 and m["recall_at_k"] == 0.5 and m["misses"] == ["Q2"]
    assert m["by_tag"]["exact-term"]["recall_at_k"] == 1.0


def test_quote_matching():
    assert quote_matches_expected("Net 30 days", "Net 30 days from invoice date.")
    assert not quote_matches_expected("Refunds take 5 days.", "Net 30 days from invoice date.")


def test_answer_metrics():
    cit = {"doc_id": "a", "chunk_id": "c:a:1", "page": 2, "quote": "Net 30 days from invoice date."}
    records = [
        rec("Q1", "answered", [cit]),
        rec(
            "Q2",
            "needs_human",
            attempts=[attempt(1, 0, ok=False), attempt(1, 0, ok=False)],
            retried=True,
            ok=False,
        ),
        rec("Q3", "abstained", attempts=[attempt(0, 0, abstained=True)]),
    ]
    m = answer_metrics(records, Q, {"input_per_mtok": 2.0, "output_per_mtok": 10.0})
    assert m["supported_answer_rate"] == 0.5
    assert m["page_accuracy"] == 1.0
    assert m["abstention_recall"] == 1.0 and m["false_abstention"] == 0.0
    assert m["unsupported_shipped"] == 0
    assert m["escalation_rate"] == round(1 / 3, 4)
    assert m["retry_rate"] == round(1 / 3, 4) and m["retry_fixed"] == 0.0
    assert m["valid_citation_rate"] == round(1 / 3, 4)
    assert m["cost_usd"] == round(3000 / 1e6 * 2 + 300 / 1e6 * 10, 4)


def test_unsupported_shipped_is_counted_if_it_ever_happens():
    m = answer_metrics(
        [rec("Q1", "answered", ok=False)], Q, {"input_per_mtok": 0, "output_per_mtok": 0}
    )
    assert m["unsupported_shipped"] == 1


def test_baseline_counts_invalid_first_attempts():
    records = [
        rec("Q1", "answered", attempts=[attempt(2, 1, ok=False), attempt(1, 1)], retried=True),
        rec("Q2", "answered"),
        rec("Q3", "abstained", attempts=[attempt(0, 0, abstained=True)]),
    ]
    assert baseline_no_guard(records) == {
        "answers_with_invalid_quote": 1,
        "of_answers": 2,
        "rate": 0.5,
        "answers_without_citations": 0,
        "answers_failing_evidence_check": 1,
        "failing_rate": 0.5,
    }


def test_misses_are_grouped_with_reasons():
    records = [
        rec("Q3", "answered", run=1),
        rec("Q3", "answered", run=2),
        rec("Q1", "abstained", retrieved=["z"]),
    ]
    misses = collect_misses(records, Q)
    assert {m["question_id"] for m in misses} == {"Q1", "Q3"}
    q3 = next(m for m in misses if m["question_id"] == "Q3")
    assert q3["runs_missed"] == [1, 2]
    q1 = next(m for m in misses if m["question_id"] == "Q1")
    assert "retrieval missed a" in q1["what_went_wrong"]


def test_percentile():
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([], 50) is None


def test_supported_needs_an_expected_document_and_the_same_numbers():
    q = {
        "expected_doc_ids": ["returns-policy"],
        "expected_quotes": ["Customers may return unused items within 30 days of delivery."],
    }
    base = {"status": "answered", "final_evidence_ok": True}
    decoy = {
        **base,
        "citations": [
            {
                "doc_id": "returns-policy-2025",
                "quote": "Customers may return unused items within 30 days of delivery.",
            }
        ],
    }
    wrong_number = {
        **base,
        "citations": [
            {
                "doc_id": "returns-policy",
                "quote": "Customers may return unused items within 14 days of delivery.",
            }
        ],
    }
    right = {
        **base,
        "citations": [{"doc_id": "returns-policy", "quote": "return unused items within 30 days"}],
    }
    from app.evaluation import is_supported

    assert not is_supported(decoy, q)
    assert not is_supported(wrong_number, q)
    assert is_supported(right, q)


def test_verify_shipped_is_independent_of_status():
    from app.evaluation import verify_shipped

    chunks = {"c:1": "Refunds are accepted within 130 days. Bulk price is 1,500 units."}
    assert verify_shipped(
        "1,500 units", [{"chunk_id": "c:1", "quote": "Bulk price is 1,500 units."}], chunks
    )
    assert not verify_shipped(
        "Within 30 days.", [{"chunk_id": "c:1", "quote": "30 days. Bulk"}], chunks
    )
    assert not verify_shipped("ok", [], chunks)
    assert not verify_shipped(
        "2,000", [{"chunk_id": "c:1", "quote": "Bulk price is 1,500 units."}], chunks
    )


def test_fully_supported_needs_every_expected_document():
    from app.evaluation import is_fully_supported, is_supported

    q = {
        "expected_doc_ids": ["a", "b"],
        "expected_quotes": [
            "Payment is due net 45 from the invoice date.",
            "A late fee of 1.5 percent per month applies.",
        ],
    }
    half = {
        "status": "answered",
        "final_evidence_ok": True,
        "citations": [{"doc_id": "b", "quote": "A late fee of 1.5 percent per month applies."}],
    }
    both = {
        **half,
        "citations": half["citations"]
        + [{"doc_id": "a", "quote": "Payment is due net 45 from the invoice date."}],
    }
    assert is_supported(half, q) and not is_fully_supported(half, q)
    assert is_fully_supported(both, q)
