"""Evidence check and the retry path, with a mocked Claude client."""

from __future__ import annotations

from dataclasses import dataclass

from app.answer import LLMResult, answer_question, build_user_prompt
from app.evidence import check_answer
from app.schemas import NOT_IN_DOCUMENTS, ModelAnswer
from tests.conftest import FakeLLM


@dataclass(frozen=True)
class C:
    chunk_id: str
    doc_id: str
    title: str
    page: int | None
    text: str


RETURNS = C(
    "hg:returns-policy:1",
    "returns-policy",
    "Returns Policy",
    None,
    "Customers may return unused items within 30 days of delivery. Return shipping is paid by the customer.",
)
WARRANTY = C(
    "hg:warranty:4",
    "warranty",
    "Warranty",
    2,
    "UV damage to tent\nfabrics is excluded after the first 12 months.",
)
CHUNKS = [RETURNS, WARRANTY]


def good(answer="Customers have 30 days from delivery.", quote=None, chunk=RETURNS):
    return {
        "answer": answer,
        "abstained": False,
        "citations": [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "quote": quote or "Customers may return unused items within 30 days of delivery.",
            }
        ],
    }


def made_up():
    return good(quote="Customers may return any item within 60 days.", answer="You have 60 days.")


# ---------- evidence check ----------


def test_valid_citation_passes_and_page_comes_from_the_chunk():
    ans = ModelAnswer.model_validate(
        good(
            "UV damage is excluded after 12 months.",
            "UV damage to tent fabrics is excluded after the first 12 months.",
            WARRANTY,
        )
    )
    res = check_answer(ans, CHUNKS)
    assert res.ok and not res.errors
    cit = res.citations[0]
    assert cit.page == 2 and cit.title == "Warranty"
    assert WARRANTY.text[cit.quote_start : cit.quote_end] == cit.quote


def test_made_up_quote_is_always_rejected():
    res = check_answer(ModelAnswer.model_validate(made_up()), CHUNKS)
    assert not res.ok
    assert any("quote not found" in e for e in res.errors)


def test_chunk_not_retrieved_is_rejected():
    data = good()
    data["citations"][0]["chunk_id"] = "hg:returns-policy-2025:0"
    res = check_answer(ModelAnswer.model_validate(data), CHUNKS)
    assert not res.ok and "not one of the retrieved chunks" in res.errors[0]


def test_doc_id_mismatch_is_rejected():
    data = good()
    data["citations"][0]["doc_id"] = "warranty"
    assert not check_answer(ModelAnswer.model_validate(data), CHUNKS).ok


def test_number_not_in_any_quote_is_rejected():
    res = check_answer(ModelAnswer.model_validate(good("Customers have 45 days.")), CHUNKS)
    assert not res.ok and any("45" in e for e in res.errors)


def test_answer_without_citations_is_rejected():
    data = {"answer": "30 days.", "abstained": False, "citations": []}
    assert not check_answer(ModelAnswer.model_validate(data), CHUNKS).ok


def test_one_bad_citation_fails_the_whole_answer():
    data = good()
    data["citations"].append({"doc_id": "warranty", "chunk_id": "hg:warranty:4", "quote": "not there at all"})
    res = check_answer(ModelAnswer.model_validate(data), CHUNKS)
    assert not res.ok and len(res.citations) == 1


def test_abstained_needs_no_citations():
    data = {"answer": NOT_IN_DOCUMENTS, "abstained": True, "citations": []}
    res = check_answer(ModelAnswer.model_validate(data), CHUNKS)
    assert res.ok and res.abstained


# ---------- answer_question: the retry path ----------


def test_first_answer_passes_no_retry():
    llm = FakeLLM([good()])
    out = answer_question(llm, "How long to return?", CHUNKS)
    assert out.status == "answered" and not out.retried and len(llm.prompts) == 1
    assert out.citations[0].page is None


def test_first_fails_second_passes():
    llm = FakeLLM([made_up(), good()])
    out = answer_question(llm, "How long to return?", CHUNKS)
    assert out.status == "answered" and out.retried
    assert "quote not found" in llm.prompts[1]  # the exact error was sent back


def test_both_fail_needs_human_and_no_answer_text():
    llm = FakeLLM([made_up(), made_up()])
    out = answer_question(llm, "How long to return?", CHUNKS)
    assert out.status == "needs_human" and out.retried
    assert out.answer is None and out.citations == [] and out.errors


def test_invalid_json_counts_as_evidence_failure_and_is_retried():
    llm = FakeLLM(["this is not json", good()])
    out = answer_question(llm, "How long to return?", CHUNKS)
    assert out.status == "answered" and out.retried
    assert "not valid JSON" in llm.prompts[1]


def test_wrong_schema_is_retried():
    llm = FakeLLM([{"answer": "30", "citations": []}, good()])
    out = answer_question(llm, "q", CHUNKS)
    assert out.status == "answered" and out.retried


def test_refused_question_returns_abstained():
    llm = FakeLLM([{"answer": NOT_IN_DOCUMENTS, "abstained": True, "citations": []}])
    out = answer_question(llm, "Do you ship to Antarctica?", CHUNKS)
    assert out.status == "abstained" and out.answer == NOT_IN_DOCUMENTS and out.citations == []


def test_api_refusal_is_not_retried():
    llm = FakeLLM([LLMResult("", "refusal")])
    out = answer_question(llm, "q", CHUNKS)
    assert out.status == "needs_human" and not out.retried and len(llm.prompts) == 1


def test_at_most_one_retry():
    llm = FakeLLM([made_up(), made_up(), good()])
    answer_question(llm, "q", CHUNKS)
    assert len(llm.prompts) == 2


def test_baseline_without_guard_ships_the_made_up_answer():
    llm = FakeLLM([made_up()])
    out = answer_question(llm, "q", CHUNKS, guard=False)
    assert out.status == "answered" and out.answer == "You have 60 days."
    assert not out.attempts[0].evidence.ok  # measured, but not enforced


def test_prompt_lists_chunk_ids_and_pages_but_model_never_supplies_page():
    prompt = build_user_prompt("q", CHUNKS)
    assert 'chunk_id="hg:warranty:4"' in prompt and "page 2" in prompt
    assert "page" not in ModelAnswer.model_fields


def test_partial_number_quote_is_rejected_by_the_evidence_check():
    chunk = C("hg:refunds:0", "refunds", "Refunds", None, "Refunds are accepted within 130 days. Bulk price is 1,500 units.")
    data = {"answer": "Refunds within 30 days.", "abstained": False,
            "citations": [{"doc_id": "refunds", "chunk_id": "hg:refunds:0", "quote": "30 days. Bulk"}]}
    assert not check_answer(ModelAnswer.model_validate(data), [chunk]).ok
