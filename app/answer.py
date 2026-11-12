"""Answering with Claude: forced schema, schema validation, evidence check, one retry.

Flow (all enforced in code; the prompt only asks):
1. Claude answers in a forced JSON schema (structured outputs, `output_config.format`).
2. The JSON is parsed and validated with pydantic. Invalid JSON or schema = an evidence failure.
3. The evidence check runs (`app.evidence`).
4. On failure: ONE retry, with the exact errors sent back to Claude.
5. Still failing: status `needs_human`, no answer text is returned as fact.

A refusal stop reason from the API is not retried (it is not an evidence failure): `needs_human`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from app.evidence import CheckedCitation, ChunkLike, EvidenceResult, check_answer
from app.schemas import MODEL_ANSWER_JSON_SCHEMA, NOT_IN_DOCUMENTS, ModelAnswer

SYSTEM_PROMPT = """You answer questions about a small business's internal documents.

Rules:
- Use ONLY the document excerpts given in the user message. Never use outside knowledge.
- Every claim in your answer must be supported by a citation. A citation names the excerpt's
  doc_id and chunk_id and copies an exact quote from that excerpt: copy the words character for
  character, do not paraphrase, shorten inside the quote, or fix typos.
- Keep the answer short (one to three sentences). Every number in the answer must appear inside
  one of your quotes.
- If the excerpts do not contain the answer, set abstained to true, set answer to
  "Not in the documents." and return no citations. Do not guess.
- Some documents are superseded or describe a different supplier. Prefer the current document and
  the one the question names; if the question is ambiguous, say which document you used.
- Answer in the JSON format requested."""


class LLMUnavailable(RuntimeError):
    """Claude is unreachable or not configured. Maps to HTTP 503."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> LLMResult: ...


class AnthropicLLM:
    """Claude via the official SDK, with structured outputs and server-side refusal fallback."""

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        api_key: str,
        model: str,
        effort: str,
        max_tokens: int,
        fallback: bool,
        base_url: str = "https://api.anthropic.com",
    ):
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key, base_url=base_url, max_retries=2, timeout=120
        )
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallback = fallback

    def complete(self, system: str, user: str) -> LLMResult:
        a = self._anthropic
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": MODEL_ANSWER_JSON_SCHEMA},
            },
        )
        try:
            if self.fallback:
                response = self._client.beta.messages.create(
                    **kwargs, betas=[self.FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = self._client.messages.create(**kwargs)
        except (a.APIConnectionError, a.RateLimitError, a.InternalServerError) as exc:
            raise LLMUnavailable(f"Claude unreachable: {type(exc).__name__}") from exc
        except a.AuthenticationError as exc:
            raise LLMUnavailable("Claude rejected the API key") from exc
        except a.APIStatusError as exc:
            raise LLMUnavailable(f"Claude API error {exc.status_code}") from exc
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        usage = response.usage
        return LLMResult(
            text=text,
            stop_reason=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


def build_user_prompt(question: str, chunks: list[ChunkLike]) -> str:
    parts = ["Document excerpts:\n"]
    for c in chunks:
        page = f", page {c.page}" if c.page is not None else ""
        parts.append(
            f'<excerpt doc_id="{c.doc_id}" chunk_id="{c.chunk_id}" title="{c.title}{page}">\n'
            f"{c.text}\n</excerpt>\n"
        )
    parts.append(f"\nQuestion: {question}")
    return "\n".join(parts)


def build_retry_prompt(first_prompt: str, previous_text: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return (
        f"{first_prompt}\n\n"
        "Your previous answer was rejected by an automatic evidence check.\n"
        f"Previous answer:\n{previous_text}\n\n"
        f"Errors:\n{listed}\n\n"
        "Answer again. Copy quotes exactly from the excerpts, cite only the chunk_ids shown, and "
        "make sure every number in the answer appears in a quote. If the excerpts do not contain "
        "the answer, abstain."
    )


@dataclass
class Attempt:
    raw_text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    parsed: ModelAnswer | None
    evidence: EvidenceResult


@dataclass
class AskOutcome:
    status: str  # answered | abstained | needs_human
    answer: str | None
    citations: list[CheckedCitation]
    retried: bool
    errors: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.attempts)

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.attempts)


def parse_model_output(text: str) -> tuple[ModelAnswer | None, list[str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, ["the response was not valid JSON"]
    try:
        return ModelAnswer.model_validate(data), []
    except ValidationError as exc:
        fields = ", ".join(sorted({".".join(map(str, e["loc"])) or "root" for e in exc.errors()}))
        return None, [f"the response did not match the schema ({fields})"]


def evaluate_attempt(raw: LLMResult, chunks: list[ChunkLike]) -> Attempt:
    """Schema validation + evidence check of one model response. Pure: no API calls."""
    if raw.stop_reason == "refusal":
        ev = EvidenceResult(ok=False, errors=["the model declined to answer (refusal)"])
        return Attempt(raw.text, raw.stop_reason, raw.input_tokens, raw.output_tokens, None, ev)
    parsed, errors = parse_model_output(raw.text)
    if parsed is None:
        if raw.stop_reason == "max_tokens":
            errors = errors + ["the response was cut off (max_tokens)"]
        ev = EvidenceResult(ok=False, errors=errors)
    else:
        ev = check_answer(parsed, chunks)
    return Attempt(raw.text, raw.stop_reason, raw.input_tokens, raw.output_tokens, parsed, ev)


def outcome_from_attempts(attempts: list[Attempt]) -> AskOutcome:
    """Decide the final status from one or two evaluated attempts."""
    last = attempts[-1]
    retried = len(attempts) > 1
    if last.evidence.ok and last.parsed is not None:
        if last.evidence.abstained:
            return AskOutcome("abstained", NOT_IN_DOCUMENTS, [], retried, [], attempts)
        return AskOutcome(
            "answered", last.parsed.answer, last.evidence.citations, retried, [], attempts
        )
    return AskOutcome("needs_human", None, [], retried, list(last.evidence.errors), attempts)


def answer_question(
    llm: LLMClient, question: str, chunks: list[ChunkLike], *, guard: bool = True
) -> AskOutcome:
    """The full answer step.

    With guard=False the first answer is shipped unchecked (used only for the baseline).
    """
    prompt = build_user_prompt(question, chunks)
    first = evaluate_attempt(llm.complete(SYSTEM_PROMPT, prompt), chunks)
    if not guard:
        status = "abstained" if first.parsed and first.parsed.abstained else "answered"
        text = first.parsed.answer if first.parsed else first.raw_text
        return AskOutcome(status, text, first.evidence.citations, False, [], [first])
    if first.evidence.ok or first.stop_reason == "refusal":
        return outcome_from_attempts([first])
    retry_prompt = build_retry_prompt(prompt, first.raw_text, first.evidence.errors)
    second = evaluate_attempt(llm.complete(SYSTEM_PROMPT, retry_prompt), chunks)
    return outcome_from_attempts([first, second])
