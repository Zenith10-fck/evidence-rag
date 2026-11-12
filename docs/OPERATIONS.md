# Operations

How to run evidence-rag as a service, what it does for you, and what it does not do.
Reference build on synthetic data; these notes describe the code in this repo, not a hosted service.

## Start and stop

```bash
cp .env.example .env              # fill API_KEY, POSTGRES_PASSWORD, ANTHROPIC_API_KEY, VOYAGE_API_KEY
docker compose up -d --build      # db + api, live mode
docker compose logs -f api        # one JSON object per line
docker compose down               # keeps the data volume
```

Compose refuses to start when a required value is missing. The API refuses to start (with a message
that names the setting) when `DATABASE_URL` or `API_KEY` is missing or `API_KEYS` is malformed.

## Health

| Route | Meaning | Use for |
|---|---|---|
| `GET /health` | the process is up | container liveness |
| `GET /ready` | the database answers and the schema exists; otherwise 503 | load balancer or orchestrator readiness |

The Docker `HEALTHCHECK` calls `/ready`. Neither route needs a key and neither calls a paid API.

## Access and limits

- Callers send `X-API-Key`. `API_KEY` is one caller named `default`; `API_KEYS=name:key,name2:key2`
  adds named callers (keys of 16+ characters). Keys are compared in constant time and never logged.
- Rotate a key: add the new one as another caller, move the client, remove the old one, restart.
- `/ask` and `/retrieve` cost money, so each caller has `ASK_PER_MINUTE` and `ASK_PER_DAY`
  (sliding minute window, day resets 00:00 UTC). Over the limit: HTTP 429 with `Retry-After`.
- Question length, PDF size and page count, and `RETRIEVAL_K` are fixed by the server.
- **The usage counters live in memory, per process.** The image runs one uvicorn worker, which
  enforces them exactly. If you run several workers or hosts, each counts alone: move the limit to a
  gateway or Redis first.

## Logs

One line per request, JSON by default (`LOG_FORMAT=text` for a terminal): `request_id`, `caller`,
`method`, `route` (the template, not the raw URL), `status`, `duration_ms`, plus counts and ids.
Question text, answers, quotes and keys are not logged. The same id is returned in the
`X-Request-ID` header. An unexpected error returns `{"detail": "internal error", "request_id": ...}`
and logs the exception type with that id; no stack trace reaches the caller.

## Backups

All state is in the Postgres volume (`evidence_rag_data`). Documents can be re-ingested from their
sources, but a dump is cheaper than re-embedding:

```bash
docker compose exec db pg_dump -U evidence -d evidence_rag -Fc > evidence_rag.dump
docker compose exec -T db pg_restore -U evidence -d evidence_rag --clean < evidence_rag.dump
```

Restore into the same embedding model and `EMBEDDING_DIM`; the API refuses to start if the stored
vector size differs.

## Schema migrations

`app/store.py` holds an ordered, append-only `MIGRATIONS` list. At startup the API takes a Postgres
advisory lock, applies every version not yet in the `schema_migrations` table, and records it, so two
instances starting together cannot migrate twice. A database with a version the code does not know
(a newer release) stops the start instead of running against the wrong schema. `/ready` is 503 until
the database is at the latest version. To change the schema, add the next numbered migration; never
edit a released one. There is no automatic rollback: take a backup first.

## Embedding model changes

Chunks from different embedding models are not comparable. Use a new `collection_id` when changing
`VOYAGE_EMBEDDING_MODEL`, ingest again, then switch callers over.

## Not covered (be honest about this before calling it hosted production)

- **TLS and public exposure:** none. Put a reverse proxy or platform load balancer in front; the
  compose file binds the API and database to 127.0.0.1.
- **Shared rate limits** across workers or hosts (see above).
- **Metrics and alerting:** logs only. No Prometheus endpoint, no tracing.
- **Load testing at scale:** not measured. The only measured numbers are the answer-quality ones in
  `results.json`.
- **Authentication per end user:** callers are API keys, not people.
