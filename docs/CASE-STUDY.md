# Case study: evidence-rag

Reference build on synthetic data. Every number below is copied from `results.json` (run of
2026-11-05, `claude-sonnet-5-5` at effort `low`, embeddings `voyage-4`, reranker `rerank-3`, k = 5,
60 questions, 1 run per mode).

## The problem

A small business has policies, supplier terms and procedures in text and PDF files. Staff ask
questions; a normal chatbot answers fluently but sometimes states things the documents do not say,
and nobody can tell which answers to trust.

## The approach

- Answers must carry exact quotes. Code checks each quote against the text that was retrieved,
  and every number in the answer must appear in a quote. The document name and page come from the
  stored chunk, not from the model.
- One retry with the exact error; then the case goes to a person (`needs_human`).
- If the documents do not contain the answer, the system says so (`abstained`).
- Retrieval compared in three modes on the same questions: vector, hybrid, hybrid + rerank.

## The result

| | vector | hybrid | hybrid + rerank |
|---|---|---|---|
| recall@5 | 1.0 | 1.0 | 1.0 |
| MRR | 0.98 | 0.9567 | 1.0 |
| supported answers | 98.0% | 100.0% | 98.0% |
| fully supported (every expected document cited) | 98.0% | 96.0% | 98.0% |
| answers shipped without proof | 0 | 0 | 0 |
| correct refusals (10 unanswerable) | 100.0% | 100.0% | 100.0% |
| cost per question (USD, list price) | 0.0063 | 0.0066 | 0.0064 |

Without the evidence check (the first answer as the model wrote it), 13 of 150 answers stated a
number that no quote contained. With the check, 0 such answers were shipped; the one retry fixed 11
of the 13, and the rest went to a person.

Misses: 4, listed in the README with the reason for each. Two are a correct answer escalated
because it repeated a number from the question (the rule is strict on purpose); two are answers
that cited only one of the two documents a question needs.

## The limits

- 60 questions on synthetic documents (a made-up shop), one language, text-based PDFs only.
- One run per mode (budget); differences of one or two questions between modes are not reliable.
- Retrieval is near the ceiling on this small corpus, so the modes are hard to separate.
- Latency is not reported here: the measurement waited for a free-tier rate limit on the
  embedding API, which makes the numbers unrepresentative.
- The numbers describe this set. A client's documents need their own test questions.
