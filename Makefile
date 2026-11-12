# Short targets for Linux, CI and fresh clones. On Windows `make` is usually not installed:
# run the same commands from README.md directly.

.PHONY: test lint db run corpus ingest eval-retrieval estimate eval plots up

test:
	uv run pytest

lint:
	uv run ruff check .

db:
	docker compose up -d db

# Live mode on the host (reads .env). Local embeddings need: uv sync --extra local
run:
	uv run --extra local uvicorn app.main:app --reload

# Docker: database + API in live mode (needs .env, see .env.example)
up:
	docker compose up --build

corpus:
	uv run python scripts/make_corpus.py
	uv run python scripts/make_pdfs.py

ingest:
	uv run --extra local python scripts/ingest_corpus.py

# Free with the local embedding model
eval-retrieval:
	uv run --extra local python scripts/eval_retrieval.py

estimate:
	uv run python scripts/measure.py --estimate

# Real API calls, costs money.
eval:
	uv run python scripts/measure.py --runs 3 --yes

plots:
	uv run python scripts/plot_results.py
