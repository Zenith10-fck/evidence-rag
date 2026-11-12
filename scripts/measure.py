"""Full measurement through the real pipeline (retrieval + Claude + evidence check).

COSTS MONEY (Claude calls). It refuses to run without --yes. Run --estimate first: it prints the
number of calls and a rough cost from the prompt sizes, without calling any API.

What it writes (never edit these by hand):
- results/records.jsonl          one line per (run, mode, question): the raw material
- results.json                   aggregated metrics per mode, baseline, misses (README quotes this)
- data/recorded/answers.json     run 1 of every mode, replayed by demo mode

It resumes: records already in results/records.jsonl for the same model are skipped, so an
interrupted run (rate limit, network) can simply be started again.

Usage:
  uv run python scripts/measure.py --estimate
  uv run python scripts/measure.py --runs 3 --yes
  uv run python scripts/measure.py --runs 1 --modes hybrid --limit 5 --yes     # small smoke run
"""

from __future__ import annotations

import argparse
import json
import time

from common import ROOT, live_settings, questions, read_results, setup, today, write_results

from app.answer import (
    SYSTEM_PROMPT,
    AnthropicLLM,
    LLMUnavailable,
    answer_question,
)
from app.corpus import chunk_corpus, load_corpus
from app.embeddings import voyage_limiter
from app.evaluation import answer_metrics, baseline_no_guard, collect_misses, verify_shipped
from app.evidence import validate_citation
from app.rerank import VoyageReranker
from app.retrieve import retrieve
from app.text import locate_quote

MODES = ("vector", "hybrid", "hybrid_rerank")
RECORDS = ROOT / "results" / "records.jsonl"
RECORDINGS = ROOT / "data" / "recorded" / "answers.json"

# USD per million tokens, from the claude-api skill's model table (cached 2026-10-06).
# Re-check https://platform.claude.com/docs/en/about-claude/pricing before quoting a cost.
PRICES = {
    "claude-sonnet-5-5": {"input_per_mtok": 2.0, "output_per_mtok": 10.0},
    "claude-haiku-5-5": {"input_per_mtok": 0.10, "output_per_mtok": 0.50},
    "claude-opus-5-5": {"input_per_mtok": 4.0, "output_per_mtok": 20.0},
}
PRICE_SOURCE = "claude-api skill model table, cached 2026-10-06 (list price, no caching discount)"


def attempt_summary(attempt, chunks) -> dict:
    by_id = {c.chunk_id: c for c in chunks}
    cits = attempt.parsed.citations if attempt.parsed else []
    valid = sum(1 for i, c in enumerate(cits, 1) if validate_citation(i, c, by_id)[1] is None)
    # For the baseline: quotes that appear in NO retrieved chunk (made up, or paraphrased).
    not_found = sum(1 for c in cits if not any(locate_quote(ch.text, c.quote) for ch in chunks))
    return {
        "raw_text": attempt.raw_text,
        "stop_reason": attempt.stop_reason,
        "parsed_ok": attempt.parsed is not None,
        "abstained": bool(attempt.parsed and attempt.parsed.abstained),
        "citations_total": len(cits),
        "citations_valid": valid,
        "quotes_not_in_retrieved": not_found,
        "evidence_ok": attempt.evidence.ok,
        "errors": attempt.evidence.errors,
    }


def run_config(settings, embedding_model: str) -> dict:
    """Everything that changes the results. Records from another config are never mixed in."""
    return {
        "claude_model": settings.answer_model,
        "answer_effort": settings.answer_effort,
        "embedding_model": embedding_model,
        "k": settings.retrieval_k,
        "chunk_max_chars": settings.chunk_max_chars,
    }


def load_done(config: dict) -> list[dict]:
    if not RECORDS.exists():
        return []
    rows = [json.loads(line) for line in RECORDS.read_text("utf-8").splitlines() if line.strip()]
    return [r for r in rows if all(r.get(k) == v for k, v in config.items())]


def estimate(settings, qs, modes, runs) -> None:
    """Rough, offline: prompt characters / 4 for input; assumed output per call."""
    docs = load_corpus(ROOT / settings.corpus_dir)
    chunks = list(chunk_corpus(docs, "x", settings.chunk_max_chars).values())
    avg_chunk = sum(len(c.text) for c in chunks) / len(chunks)
    prompt_chars = len(SYSTEM_PROMPT) + settings.retrieval_k * (avg_chunk + 120) + 200
    calls = len(qs) * len(modes) * runs
    calls_with_retries = int(calls * 1.15)  # assume up to 15% need the one retry
    in_tok = calls_with_retries * prompt_chars / 4
    assumed_out = 600  # answer JSON plus adaptive thinking at low effort; an assumption
    out_tok = calls_with_retries * assumed_out
    price = PRICES.get(settings.answer_model)
    print(f"model {settings.answer_model}, effort {settings.answer_effort}")
    print(f"calls: {calls} (+ up to ~15% retries = {calls_with_retries})")
    print(
        f"input tokens ~{in_tok:,.0f}, output tokens ~{out_tok:,.0f} (assumes {assumed_out}/call)"
    )
    if price:
        cost = in_tok / 1e6 * price["input_per_mtok"] + out_tok / 1e6 * price["output_per_mtok"]
        print(f"rough cost ~{cost:.2f} USD at {price} ({PRICE_SOURCE})")
    print("This is an estimate, not a measurement. Nothing was called.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke run)")
    ap.add_argument("--estimate", action="store_true", help="print a cost estimate and exit")
    ap.add_argument("--yes", action="store_true", help="confirm that this run costs money")
    ap.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="hard spending limit for THIS invocation (list price from token counts); stops\n"
        "before a call that could cross it. Re-run to resume.",
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="no API calls: rebuild results.json from results/records.jsonl",
    )
    args = ap.parse_args()

    settings = live_settings()
    qs = questions()[: args.limit] if args.limit else questions()
    modes = [m for m in args.modes.split(",") if m]
    if args.estimate:
        estimate(settings, qs, modes, args.runs)
        return
    if args.aggregate_only:
        aggregate_only(settings, qs, modes, args.runs)
        return
    if not args.yes:
        raise SystemExit("This run calls Claude and costs money. Run --estimate, then add --yes.")
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set in .env.")
    if "hybrid_rerank" in modes and not settings.voyage_api_key:
        print("No VOYAGE_API_KEY: skipping hybrid_rerank (the reranker is an API).")
        modes = [m for m in modes if m != "hybrid_rerank"]

    store, embedder, collection = setup(settings)
    llm = AnthropicLLM(
        settings.anthropic_api_key,
        settings.answer_model,
        settings.answer_effort,
        settings.answer_max_tokens,
        settings.answer_refusal_fallback,
        settings.answer_api_base_url,
    )
    reranker = (
        VoyageReranker(
            settings.voyage_api_key,
            settings.rerank_model,
            voyage_limiter(settings.voyage_api_key, settings.voyage_rpm, settings.voyage_tpm),
        )
        if settings.voyage_api_key
        else None
    )

    config = run_config(settings, embedder.model)
    done = load_done(config)
    done_keys = {(r["run"], r["mode"], r["question_id"]) for r in done}
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    total = len(qs) * len(modes) * args.runs
    n = 0
    price = PRICES.get(settings.answer_model)
    if args.budget_usd is not None and price is None:
        raise SystemExit(f"No price known for {settings.answer_model}; cannot enforce a budget.")
    spent = 0.0
    calls_cost: list[float] = []
    with RECORDS.open("a", encoding="utf-8", newline="\n") as out:
        for run in range(1, args.runs + 1):
            for mode in modes:
                for q in qs:
                    n += 1
                    if (run, mode, q["id"]) in done_keys:
                        continue
                    if args.budget_usd is not None:
                        # worst case for the next question: twice the most expensive so far
                        # (a retry doubles it); before any data, assume 0.05 USD
                        worst = 2 * max(calls_cost) if calls_cost else 0.05
                        if spent + worst > args.budget_usd:
                            print(
                                f"Budget stop at {n}/{total}: spent {spent:.4f} USD, next "
                                f"question could cost up to {worst:.4f}. Records are saved; "
                                "re-run to resume, or use --aggregate-only."
                            )
                            return
                    started = time.perf_counter()
                    res = retrieve(
                        store,
                        embedder,
                        collection_id=collection,
                        question=q["question"],
                        mode=mode,
                        k=settings.retrieval_k,
                        candidates=settings.rerank_candidates,
                        reranker=reranker if mode == "hybrid_rerank" else None,
                    )
                    try:
                        outcome = answer_question(llm, q["question"], res.chunks)
                    except LLMUnavailable as exc:
                        msg = f"Stopped at {n}/{total}: {exc}. Re-run to resume."
                        raise SystemExit(msg) from exc
                    shipped = [
                        {
                            "doc_id": c.doc_id,
                            "chunk_id": c.chunk_id,
                            "page": c.page,
                            "quote": c.quote,
                        }
                        for c in outcome.citations
                    ]
                    record = {
                        **config,
                        "date": today(),
                        "question_id": q["id"],
                        "type": q["type"],
                        "tags": q["tags"],
                        "mode": mode,
                        "run": run,
                        "question": q["question"],
                        "rerank": res.rerank,
                        "retrieved_chunk_ids": [c.chunk_id for c in res.chunks],
                        "retrieved_doc_ids": [c.doc_id for c in res.chunks],
                        "status": outcome.status,
                        "retried": outcome.retried,
                        "answer": outcome.answer,
                        "errors": outcome.errors,
                        "citations": shipped,
                        "attempts": [attempt_summary(a, res.chunks) for a in outcome.attempts],
                        # re-checked from the shipped parts, separately from the status
                        "final_evidence_ok": verify_shipped(
                            outcome.answer, shipped, {c.chunk_id: c.text for c in res.chunks}
                        )
                        if outcome.status == "answered"
                        else True,
                        "latency_s": round(time.perf_counter() - started, 3),
                        "input_tokens": outcome.input_tokens,
                        "output_tokens": outcome.output_tokens,
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    done.append(record)
                    cost = 0.0
                    if price:
                        cost = (
                            outcome.input_tokens / 1e6 * price["input_per_mtok"]
                            + outcome.output_tokens / 1e6 * price["output_per_mtok"]
                        )
                    spent += cost
                    calls_cost.append(cost)
                    print(
                        f"[{n}/{total}] run {run} {mode:13} {q['id']} -> {outcome.status}"
                        f"{' (retried)' if outcome.retried else ''}  "
                        f"tokens {outcome.input_tokens}/{outcome.output_tokens}  "
                        f"spent {spent:.4f} USD"
                    )

    selected = [r for r in done if r["run"] <= args.runs and r["mode"] in modes]
    aggregate(settings, embedder.model, selected, qs, args.runs)


def aggregate_only(settings, qs: list[dict], modes: list[str], runs: int) -> None:
    """Rebuild results.json and the recordings from saved records, without any API call."""
    embedder_model = (
        settings.voyage_embedding_model
        if settings.resolved_embedding_provider == "voyage"
        else settings.local_embedding_model
    )
    done = load_done(run_config(settings, embedder_model))
    selected = [r for r in done if r["run"] <= runs and r["mode"] in modes]
    if not selected:
        raise SystemExit("No saved records for this configuration.")
    present_runs = max(r["run"] for r in selected)
    aggregate(settings, embedder_model, selected, qs, present_runs)


def aggregate(settings, embedding_model: str, records: list[dict], qs: list[dict], runs: int):
    by_id = {q["id"]: q for q in qs}
    records = [r for r in records if r["question_id"] in by_id]
    price = PRICES.get(settings.answer_model, {"input_per_mtok": 0.0, "output_per_mtok": 0.0})
    modes_present = [m for m in MODES if any(r["mode"] == m for r in records)]
    results = read_results()
    results.update(
        {
            "date": today(),
            "claude_model": settings.answer_model,
            "claude_effort": settings.answer_effort,
            "embedding_model": embedding_model,
            "rerank_model": settings.rerank_model if "hybrid_rerank" in modes_present else None,
            "k": settings.retrieval_k,
            "runs": runs,
            "question_count": len(qs),
            "price": {**price, "source": PRICE_SOURCE},
            "modes": {},
            "per_run": {},
        }
    )
    for mode in modes_present:
        mode_records = [r for r in records if r["mode"] == mode]
        results["modes"][mode] = answer_metrics(mode_records, by_id, price)
        results["per_run"][mode] = {
            str(run): answer_metrics([r for r in mode_records if r["run"] == run], by_id, price)
            for run in sorted({r["run"] for r in mode_records})
        }
    run1 = [r for r in records if r["run"] == 1]
    results["baseline_no_guard"] = {
        **baseline_no_guard(run1),
        "definition": "first attempt of each run-1 answer as the model wrote it, before the "
        "evidence check and retry (same prompt, same model); all modes",
    }
    results["misses"] = collect_misses(records, by_id)
    write_results(results)

    # Raw model output is stored unchanged. Demo mode chunks the corpus under the same
    # collection id, so the recorded chunk ids match without any rewriting.
    recordings = {
        "source": f"scripts/measure.py run 1, {today()}, {settings.answer_model}",
        "embedding_model": embedding_model,
        "collection_id": run1[0]["retrieved_chunk_ids"][0].split(":", 1)[0] if run1 else "",
        "chunk_max_chars": settings.chunk_max_chars,
        "recordings": [
            {
                "question_id": r["question_id"],
                "question": r["question"],
                "mode": r["mode"],
                "retrieved_chunk_ids": r["retrieved_chunk_ids"],
                "attempts": [
                    {"raw_text": a["raw_text"], "stop_reason": a["stop_reason"]}
                    for a in r["attempts"]
                ],
            }
            for r in run1
        ],
    }
    RECORDINGS.parent.mkdir(parents=True, exist_ok=True)
    RECORDINGS.write_text(json.dumps(recordings, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {RECORDINGS.name} ({len(run1)} recordings)")


if __name__ == "__main__":
    main()
