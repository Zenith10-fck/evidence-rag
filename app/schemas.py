"""Pydantic models: API requests and responses, and the schema Claude must answer with."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
RetrievalMode = Literal["vector", "hybrid", "hybrid_rerank"]
AskStatus = Literal["answered", "abstained", "needs_human"]
NOT_IN_DOCUMENTS = "Not in the documents."
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


# ---------- the schema Claude answers with (untrusted until checked) ----------


class ModelCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    chunk_id: str
    quote: str = Field(description="An exact sentence or phrase copied from that chunk.")


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    abstained: bool
    citations: list[ModelCitation]


# JSON schema sent to Claude with output_config.format (structured outputs).
MODEL_ANSWER_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstained": {"type": "boolean"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["doc_id", "chunk_id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "abstained", "citations"],
    "additionalProperties": False,
}


# ---------- documents ----------


class DocumentIn(BaseModel):
    collection_id: str = Field(default="default", pattern=ID_PATTERN)
    doc_id: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1, max_length=200)
    version: str = Field(default="", max_length=64)
    text: str = Field(min_length=1, max_length=500_000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_id": "harbor-goods",
                "doc_id": "returns-policy",
                "title": "Returns Policy",
                "version": "2026-01",
                "text": "Customers may return unused items within 30 days of delivery.",
            }
        }
    )


class IngestOut(BaseModel):
    collection_id: str
    doc_id: str
    status: Literal["created", "updated", "unchanged"]
    chunks: int
    pages: int | None = None


class DocumentInfo(BaseModel):
    collection_id: str
    doc_id: str
    title: str
    version: str
    source_type: Literal["text", "pdf"]
    pages: int | None
    chunks: int


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


# ---------- ask ----------


class AskIn(BaseModel):
    question: Question
    collection_id: str = Field(default="default", pattern=ID_PATTERN)
    mode: RetrievalMode = "hybrid"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "How many days do customers have to return unused items?",
                "collection_id": "harbor-goods",
                "mode": "hybrid",
            }
        }
    )


class CitationOut(BaseModel):
    doc_id: str
    title: str
    chunk_id: str
    page: int | None = Field(description="Copied from the stored chunk, never from the model.")
    quote: str
    quote_start: int = Field(description="Start of the quote inside chunk_text.")
    quote_end: int = Field(description="End of the quote inside chunk_text.")
    chunk_text: str = Field(description="The retrieved chunk, so a UI can highlight the quote.")


class AskOut(BaseModel):
    status: AskStatus
    answer: str | None
    citations: list[CitationOut]
    retried: bool
    mode: RetrievalMode
    errors: list[str] = []
    rerank: Literal["applied", "skipped"] | None = None
    latency_ms: int | None = None
    run_mode: Literal["live", "demo"] = "live"


# ---------- retrieve (debug / demo of retrieval without Claude) ----------


class RetrieveIn(BaseModel):
    question: Question
    collection_id: str = Field(default="default", pattern=ID_PATTERN)
    mode: RetrievalMode = "hybrid"


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    page: int | None
    score: float
    text: str


class RetrieveOut(BaseModel):
    mode: RetrievalMode
    rerank: Literal["applied", "skipped"] | None = None
    chunks: list[RetrievedChunkOut]


class HealthOut(BaseModel):
    status: Literal["ok"]
    mode: Literal["live", "demo"]


class ReadyOut(BaseModel):
    status: Literal["ready"]
    mode: Literal["live", "demo"]
    database: Literal["ok", "not_used"]
