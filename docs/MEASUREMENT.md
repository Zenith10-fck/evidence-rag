# Measurement

Two scripts, both write into `results.json` (never typed by hand; the README may quote only that file):

- `uv run python scripts/eval_retrieval.py` — retrieval only (recall@k, MRR, per mode). Calls only the
  embedding API (and the reranker for `hybrid_rerank`). Small cost.
- `uv run python scripts/measure.py --runs 3 --yes` — the full `/ask` pipeline with Claude. **Costs
  money; the script refuses to run without `--yes`.**

Then `uv run python scripts/plot_results.py` draws the charts into `results/` from `results.json`.

## What is measured

| Measure | Definition |
|---|---|
| Recall@k | share of answerable questions where every `expected_doc_ids` document is among the top-k retrieved chunks |
| MRR | mean of 1 / rank of the first chunk from an expected document |
| Valid citation rate | share of citations whose quote is an exact substring (whitespace-normalised) of the cited retrieved chunk |
| Page accuracy | for PDF questions: share of answered questions whose cited page equals an `expected_pages` entry |
| Supported-answer rate | share of answerable questions answered with all citations valid and at least one expected quote cited |
| Unsupported shipped | answers with `status=answered` but a failed evidence check. **Must be 0**: the guard makes it impossible; the measure proves it |
| Abstention recall | share of `unanswerable` questions that returned `abstained` |
| False abstention | share of answerable questions wrongly `abstained` |
| Escalation rate | share of questions that ended `needs_human` |
| Retry rate | share of questions that needed the one retry, and how many of those the retry fixed |
| Latency | p50 and p95 of `/ask` end to end |
| Cost | from token counts at the list price on the run date (read the price from current docs) |

Retrieval modes compared on the same questions: `vector`, `hybrid`, `hybrid_rerank`.
Runs: 3 per question; report each run and the mean. Report each question tag
(`exact-term`, `paraphrase`) separately from the totals.

Also report a **baseline** so a reader can see what the guard buys: the same questions answered
with the evidence check switched off (answer shipped as the model wrote it), and how many of
those answers contain a quote that is not in the retrieved text. This is the "hallucination
reduction" number clients ask about. It must be labelled clearly as a baseline, not as the product's behaviour.

Implementation (decided 2026-11-04): instead of a separate run, the baseline uses the FIRST
attempt of every run-1 answer, as the model wrote it, before the evidence check and retry (same
prompt, same model, so it is the same thing as a run with the check switched off, without paying
twice). A baseline answer counts as invalid only if one of its quotes appears in no retrieved
chunk; answers with no citation are reported separately (`answers_without_citations`).

`unsupported_shipped` is computed by re-checking each shipped answer from its parts
(`app.evaluation.verify_shipped`), separately from the status the pipeline set.

`supported_answer_rate` counts a citation as an expected quote only if it comes from one of the
expected documents and either contains (or is contained in) an expected quote, or shares at
least 80% of its words with it and carries exactly the same numbers.

## `results.json` format

```json
{
  "date": "YYYY-MM-DD",
  "claude_model": "<id from the claude-api skill>",
  "embedding_model": "<voyage model id>",
  "rerank_model": "<voyage rerank model id, or null if not used>",
  "k": 5,
  "runs": 3,
  "question_count": 60,
  "modes": {
    "vector":        { "recall_at_k": null, "mrr": null, "valid_citation_rate": null,
                       "page_accuracy": null, "supported_answer_rate": null,
                       "unsupported_shipped": null, "abstention_recall": null,
                       "false_abstention": null, "escalation_rate": null, "retry_rate": null,
                       "latency_p50_s": null, "latency_p95_s": null, "cost_usd": null,
                       "by_tag": { "exact-term": {}, "paraphrase": {} } },
    "hybrid":        { "...": "same fields" },
    "hybrid_rerank": { "...": "same fields" }
  },
  "baseline_no_guard": { "answers_with_invalid_quote": null, "of_answers": null },
  "misses": [ { "question_id": "Q000", "mode": "hybrid", "what_went_wrong": "" } ]
}
```

All values are `null` until a real run fills them. Nothing is typed in by hand.

## Honest reporting

- List every miss in the README, with the question id and the reason, even if the cause is a
  weak answer key and not the model.
- Do not change the prompt or the answer keys to improve a number after seeing the results unless
  the change is also described in the README (what changed, why, before and after).
- If reranking or hybrid does not beat the simpler mode on this question set, say so. A clear
  "rerank did not help here, and this is why we think so" is more credible than a hidden result.
- State the limits next to the numbers: 60 questions, synthetic documents, one language, text-based
  PDFs only, one model. The numbers describe this set, not every RAG system.
- Tests prove the code, only `results.json` speaks for the model.
