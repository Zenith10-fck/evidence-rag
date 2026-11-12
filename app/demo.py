"""Demo mode: replay recorded answers, with no database and no API keys.

- Recordings (`data/recorded/answers.json`) are written only by `scripts/measure.py` from a real
  run. Each recording holds the question, the mode, the retrieved chunk ids and the model's raw
  answer text for each attempt. Nothing in it is written by hand.
- The corpus is loaded from `data/corpus/` into memory and chunked with the same code as the live
  ingest, so the chunk ids match.
- Each replay goes through the same schema validation and evidence check as a live answer.
- A question that is not recorded returns a clear message, never a made-up answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.answer import AskOutcome, LLMResult, evaluate_attempt, outcome_from_attempts
from app.corpus import MemoryChunk, chunk_corpus, load_corpus
from app.text import normalize_ws

NOT_RECORDED = (
    "demo mode: this question is not recorded. Pick one of the recorded questions, or run the "
    "service in live mode with API keys."
)


def _key(question: str, mode: str) -> tuple[str, str]:
    return normalize_ws(question).lower(), mode


@dataclass(frozen=True)
class Recording:
    question: str
    mode: str
    question_id: str | None
    retrieved_chunk_ids: list[str]
    attempts: list[dict]  # {"raw_text", "stop_reason"}


class DemoBackend:
    def __init__(self, corpus_dir: str, recordings_path: str, collection_id: str, chunk_chars: int):
        data: dict = {}
        path = Path(recordings_path)
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
        self.meta: dict = {k: v for k, v in data.items() if k != "recordings"}
        # Chunk ids carry the collection id used in the real run; use the same one.
        self.collection_id = data.get("collection_id") or collection_id
        self.docs = load_corpus(corpus_dir) if Path(corpus_dir).exists() else []
        # Same chunk size as the recorded run, so the chunk texts behind the ids are identical.
        chunk_chars = int(data.get("chunk_max_chars") or chunk_chars)
        self.chunks: dict[str, MemoryChunk] = chunk_corpus(
            self.docs, self.collection_id, chunk_chars
        )
        self.accepted_collections = {"default", collection_id, self.collection_id}
        self.recordings: dict[tuple[str, str], Recording] = {}
        for r in data.get("recordings", []):
            rec = Recording(
                question=r["question"],
                mode=r["mode"],
                question_id=r.get("question_id"),
                retrieved_chunk_ids=r["retrieved_chunk_ids"],
                attempts=r["attempts"],
            )
            self.recordings[_key(rec.question, rec.mode)] = rec

    def recorded_questions(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for rec in self.recordings.values():
            entry = seen.setdefault(
                rec.question, {"question": rec.question, "id": rec.question_id, "modes": []}
            )
            entry["modes"].append(rec.mode)
        return sorted(seen.values(), key=lambda e: (e["id"] or "", e["question"]))

    def ask(self, question: str, mode: str) -> AskOutcome | None:
        rec = self.recordings.get(_key(question, mode))
        if rec is None:
            return None
        missing = [cid for cid in rec.retrieved_chunk_ids if cid not in self.chunks]
        if missing:
            return AskOutcome(
                "needs_human",
                None,
                [],
                False,
                [f"recording does not match the corpus (unknown chunk {missing[0]})"],
            )
        chunks = [self.chunks[cid] for cid in rec.retrieved_chunk_ids]
        attempts = [
            evaluate_attempt(LLMResult(a["raw_text"], a.get("stop_reason")), chunks)
            for a in rec.attempts
        ]
        return outcome_from_attempts(attempts)
