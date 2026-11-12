# Architecture

## Flow

```
ingest:  text or PDF ──► pages ──► chunks (never cross a page, keep page no.)
                     ──► Voyage embeddings ──► Postgres (documents, chunks, vector, tsvector)

ask:     question ──► guards (key, length, collection)
                  ──► retrieve top-k  (mode: vector | hybrid | hybrid_rerank)
                        hybrid = vector + full-text merged by reciprocal rank fusion
                        hybrid_rerank = hybrid top 20, reordered by the Voyage reranker, keep k
                  ──► Claude, forced schema: answer + citations + abstained
                  ──► schema check (pydantic)
                  ──► evidence check (each quote ⊂ a retrieved chunk, whitespace-normalised only)
                        ├─ all pass ──► status "answered"  (or "abstained" if abstained=true and no citations)
                        ├─ some fail ──► one retry with the exact errors ──► check again
                        └─ still fail ──► status "needs_human" (no answer text is returned as fact)
```

## Tables (Postgres 16 with pgvector)

- `documents(collection_id text, doc_id text, title text, version text, source_type text
  ('text' | 'pdf'), content_hash text, created_at, primary key (collection_id, doc_id))`
- `chunks(chunk_id text primary key, collection_id text, doc_id text, position int,
  page int null, text text, embedding vector(<dim>), tsv tsvector,
  foreign key (collection_id, doc_id) references documents)`
  - `page` is null for plain text documents and 1-based for PDFs. `chunk_id` looks like
    `<collection_id>:<doc_id>:<position>`.
  - `<dim>` is the embedding dimension of the chosen Voyage model: read it from the current Voyage
    docs, do not guess. (Check on 4 Nov 2026: the voyage-4 family defaults to 1024 and offers
    256, 512, 1024, 2048. Re-check before writing the migration.)
  - Indexes: vector index (HNSW or IVFFlat, choose and note why), a GIN index on `tsv`, and a
    b-tree on `(collection_id, doc_id)`.
- Every retrieval query filters by `collection_id` in SQL. A test must prove that a question in
  collection A never returns a chunk from collection B.

Re-ingesting a document with the same `content_hash` does nothing (`unchanged`); a changed hash
replaces its chunks inside one transaction.

## PDF handling

- Extract text per page with a pure-Python PDF library.
- A chunk never spans two pages, so a citation points to exactly one page.
- Text-based PDFs only. Scanned PDFs (OCR) are out of scope and rejected with a clear message when
  a PDF has no extractable text.
- Quotes are checked against the extracted text, after whitespace normalisation (PDF line breaks
  are collapsed to single spaces). Hyphenation or ligature differences are listed as known limits,
  not fixed silently.

## Guards (all in code)

| Guard | Rule |
|---|---|
| API key | header `X-API-Key` compared in constant time with every configured caller key (`API_KEY` is caller `default`; `API_KEYS=name:key,...` adds named callers); missing or wrong = 401 |
| Usage limits | `/ask` and `/retrieve` (they cost money): per caller, per minute (sliding window) and per day (resets 00:00 UTC); over a limit = 429 with `Retry-After`; a refused request is not counted; wrong-key requests use nobody's quota. In demo mode the limit is per client IP. State is in memory and per process (one worker enforces exactly; with several workers or hosts use a gateway or Redis) |
| Config | a malformed `API_KEYS` (bad format, duplicate name or key, key under 16 characters) stops the start instead of locking a caller out |
| CORS | off unless `CORS_ORIGINS` lists origins; only `GET`/`POST` and the `X-API-Key` header are allowed |
| Question length | 1 to 500 characters, else 422 |
| `doc_id`, `collection_id` | letters, digits, `-` and `_`, up to 64 characters |
| PDF upload | content type and file signature checked, size limit set on the server, page limit set on the server |
| Collection | taken from the request, defaults to `default`; every query is filtered by it in SQL |
| Retrieval size | `k` is set on the server, not by the caller |
| Retry | at most one, only for evidence failures |
| Refusal | if no retrieved chunk supports an answer, the answer is `abstained`, never a guess |
| Logging | one JSON line per request (`LOG_FORMAT=text` for a terminal): request id (also in the `X-Request-ID` response header), caller name, route template, status, duration; for `/ask` also answer status, retried, citation count, token counts and the question's length. Never logged: keys, question text, answers, document text, raw URLs of unknown routes (`app/logs.py`) |

## Evidence check, precisely

Input: the model's `citations[{doc_id, chunk_id, quote}]` and the chunks that were retrieved.
A citation is valid only if: `chunk_id` is among the retrieved chunks, `doc_id` matches that chunk,
and `quote` (after collapsing runs of whitespace) is a substring of the chunk text (same
normalisation). Every sentence of `answer` that states a fact must be covered by at least one
valid citation; an answer with zero valid citations and `abstained=false` fails.

For a valid citation, the code copies `title`, `page` and the position of the quote inside the chunk
from the stored chunk into the response. The model never supplies `page`. The `/ui` page uses the
position to highlight the quote in its source text.

How "every sentence is covered" is checked (`app/evidence.py`):

1. An answer that is not abstained must carry at least one citation, and EVERY citation must be
   valid (one bad citation fails the whole answer, which then gets the one retry).
2. Every number written in the answer (e.g. `30`, `8.40`, `1,500`) must appear in at least one valid
   cited quote (thousands separators are ignored when comparing).
3. A quote must be at least 8 characters after whitespace normalisation.
4. A quote must not start or end inside a word or a number: "30 days" does not match inside
   "130 days", and "500 units" does not match inside "1,500 units" (found by an independent
   code review; otherwise a partial number could pass the number rule).

Limit, stated plainly: a sentence without a number that the quotes do not support is not caught by
code. The measurement reports how often answers cite an expected quote (`supported_answer_rate`)
and how often they contain the key facts of the answer key, to show how well this works in practice.

## Error cases

| Case | Result |
|---|---|
| Voyage or Claude unreachable | 503 with a short message, nothing stored |
| Reranker unreachable | fall back to `hybrid` order, set `"rerank": "skipped"` in the response, never fail the question |
| PDF with no extractable text, too large, or too many pages | 422 with the reason |
| Claude returns invalid schema | counts as an evidence failure, one retry |
| No documents ingested | `/ask` returns 409 "no documents" |
| Question about something not in the documents | `abstained` with `answer = "Not in the documents."` |

## Demo mode

If `EVIDENCE_RAG_MODE=demo` (or no keys), `/ask` replays recorded answers for the known question
set and still runs the evidence check against the corpus.

- Recordings (`data/recorded/answers.json`) come from the real measurement run, never written
  by hand. A recording stores the question, the retrieved chunk ids and the model's raw answer.
- In demo mode the corpus is loaded from `data/corpus/` into memory, so no Postgres, Voyage or
  Anthropic call is needed. A question that is not in the recordings returns a clear
  "demo mode: this question is not recorded" message, never a made-up answer.
- The response includes `"mode": "demo"` and `/ui` shows a visible "demo mode" badge.

## Decisions recorded during the build (2026-11-04)

| Topic | Decision | Why |
|---|---|---|
| Embedding dimension | 1024 | default of the voyage-4 family; the local model `BAAI/bge-large-en-v1.5` also has 1024 |
| Embedding input | title, a line break, then the chunk text | the title gives a chunk its context; the stored text (used for quotes) stays pure |
| Vector index | HNSW, cosine | no training step (IVFFlat must be built after data exists); `SET hnsw.iterative_scan = relaxed_order` so the collection filter still returns k rows |
| Full-text | `to_tsvector('english', title + text)`, GIN index; query = OR of the question's alphanumeric terms, ranked by `ts_rank_cd` | AND of all terms almost never matches a natural-language question; only `[A-Za-z0-9]` tokens reach `to_tsquery`, so no operator can be injected |
| Fusion | reciprocal rank fusion, k = 60, 20 candidates per side | standard, no score normalisation needed |
| Reranker | Voyage `rerank-3` on the hybrid top 20 | current recommended model in the Voyage docs (checked 2026-11-04) |
| PDF library | pypdf | pure Python, maintained, per-page `extract_text()` is enough for text PDFs |
| Chunking | sentence units, max 700 characters, one-sentence overlap, never across a page, headings glued to the next unit | keeps quotes whole; tested on every answer-key quote |
| Claude call | `claude-sonnet-5-5`, effort `low`, structured outputs via `output_config.format`, server-side refusal fallback (beta `server-side-fallback-2026-07-01`, `fallbacks="default"`) | low effort is enough for extraction-style answers; a refusal is not retried and becomes `needs_human` |
| Retry prompt | a fresh single user message: the original prompt, the previous answer and the exact errors | no multi-turn history, so nothing from the first turn has to be replayed |
| Transactions | psycopg connections in autocommit mode; each write is an explicit `conn.transaction()` | without autocommit a transaction block is only a savepoint inside an implicit transaction and is rolled back on close (found by a failing test) |
| Settings names | `ANSWER_MODEL`, `ANSWER_EFFORT`, ...; Anthropic `base_url` set explicitly | `CLAUDE_EFFORT` and `ANTHROPIC_BASE_URL` set by other tools in the shell were being picked up |
| Local embeddings | `BAAI/bge-large-en-v1.5`, pinned revision, safetensors only, no remote code | free retrieval measurement; `voyage-4-nano` needs `trust_remote_code` and is opt-in |
| Demo recordings | stored exactly as the model wrote them; demo mode chunks the corpus under the recorded collection id | nothing in a recording is edited, so the replay proves the real run |
