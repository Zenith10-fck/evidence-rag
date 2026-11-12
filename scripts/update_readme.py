"""Rewrite the "Measured results" block of README.md from results.json.

The block between `<!-- results:start -->` and `<!-- results:end -->` is generated: every number in
it is read from results.json, nothing is typed by hand. Run after measure.py / eval_retrieval.py.

Usage: uv run python scripts/update_readme.py
"""

from __future__ import annotations

import re

from common import ROOT, read_results

README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
MODES = ("vector", "hybrid", "hybrid_rerank")
LABEL = {"vector": "vector", "hybrid": "hybrid", "hybrid_rerank": "hybrid + rerank"}


def fmt(v, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and pct:
        return f"{v * 100:.1f}%"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def retrieval_block(results: dict) -> list[str]:
    out = ["### Retrieval", ""]
    for model, block in results.get("retrieval_eval", {}).items():
        out += [
            f"Embeddings `{model}`, measured {block['date']}: {block['questions']} answerable "
            f"questions, k = {block['k']}"
            + (f", reranker `{block['rerank_model']}`." if block.get("rerank_model") else "."),
            "",
            "| Mode | recall@k | hit@1 | MRR | misses |",
            "|---|---|---|---|---|",
        ]
        for mode in MODES:
            m = block["modes"].get(mode)
            if not m:
                continue
            if "recall_at_k" not in m:
                out.append(f"| {LABEL[mode]} | not measured: {m.get('not_measured', '')} | | | |")
                continue
            misses = ", ".join(m["misses"]) or "none"
            out.append(
                f"| {LABEL[mode]} | {fmt(m['recall_at_k'])} | {fmt(m.get('hit_at_1'))} | "
                f"{fmt(m['mrr'])} | {misses} |"
            )
        chart = ROOT / "results" / f"retrieval_{model.split('/')[-1]}.png"
        if chart.exists():
            out += ["", f"![Retrieval, {model}](results/{chart.name})"]
        out.append("")
    return out


def answers_block(results: dict) -> list[str]:
    modes = [m for m in MODES if m in results.get("modes", {})]
    if not modes:
        return ["### Answers", "", "Not measured yet.", ""]
    expected = results["question_count"] * results["runs"]
    out = [
        "### Answers",
        "",
        f"Measured {results['date']} with `{results['claude_model']}` (effort "
        f"`{results.get('claude_effort', 'n/a')}`), embeddings `{results['embedding_model']}`, "
        f"k = {results['k']}, {results['question_count']} questions, {results['runs']} run(s) "
        f"per mode. Cost uses the list price in results.json ({results['price']['source']}).",
        "",
    ]
    if results["runs"] < 3:
        out += [
            f"Only {results['runs']} run per mode (docs/MEASUREMENT.md plans 3): the run was "
            "limited to a 1.50 USD budget. Run-to-run variation is therefore not measured.",
            "",
        ]
    header = "| | " + " | ".join(LABEL[m] for m in modes) + " |"
    out += [header, "|---|" + "---|" * len(modes)]
    rows = [
        ("questions answered by the pipeline", lambda m: f"{m['records']} of {expected}"),
        (
            "supported answers (cited an expected quote)",
            lambda m: fmt(m["supported_answer_rate"], True),
        ),
        (
            "fully supported (an expected quote from every expected document)",
            lambda m: fmt(m.get("fully_supported_answer_rate"), True),
        ),
        ("key facts in the answer", lambda m: fmt(m["key_facts_in_answer_rate"], True)),
        ("valid citations (all attempts)", lambda m: fmt(m["valid_citation_rate"], True)),
        ("page accuracy (PDF questions)", lambda m: fmt(m["page_accuracy"], True)),
        ("**answers shipped without proof**", lambda m: f"**{m['unsupported_shipped']}**"),
        ("correct refusals (unanswerable)", lambda m: fmt(m["abstention_recall"], True)),
        ("false refusals (answerable)", lambda m: fmt(m["false_abstention"], True)),
        ("escalated (needs_human)", lambda m: fmt(m["escalation_rate"], True)),
        ("needed the one retry", lambda m: fmt(m["retry_rate"], True)),
        ("retry fixed it", lambda m: fmt(m["retry_fixed"], True)),
        (
            "latency p50 / p95 (s)",
            lambda m: f"{fmt(m['latency_p50_s'])} / {fmt(m['latency_p95_s'])}",
        ),
        ("cost, total (USD)", lambda m: fmt(m["cost_usd"])),
        ("cost per question (USD)", lambda m: fmt(round(m["cost_usd"] / m["records"], 4))),
    ]
    for label, f in rows:
        out.append(f"| {label} | " + " | ".join(f(results["modes"][m]) for m in modes) + " |")
    partial = [LABEL[m] for m in modes if results["modes"][m]["records"] < expected]
    if partial:
        out += [
            "",
            f"Partial: {', '.join(partial)} stopped at the budget limit, so its rates cover only "
            "the questions it reached (the first ones in the set) and are not directly comparable.",
        ]
    base = results.get("baseline_no_guard", {})
    if base.get("of_answers"):
        out += [
            "",
            f"**Baseline without the evidence check:** {base['answers_with_invalid_quote']} of "
            f"{base['of_answers']} answers ({fmt(base['rate'], True)}) contained a quote that is "
            f"not in any retrieved chunk; {base.get('answers_without_citations', 0)} had no "
            "citation. Counting every reason the check rejects (also numbers in the answer that "
            f"no quote contains), {base.get('answers_failing_evidence_check', 'n/a')} of "
            f"{base['of_answers']} first answers ({fmt(base.get('failing_rate'), True)}) would "
            "have shipped without proof. With the check, the number shipped is the row in bold "
            f"above. ({base['definition']}.)",
        ]
    by_tag_rows = []
    for tag in ("exact-term", "paraphrase"):
        cells = []
        for m in modes:
            t = results["modes"][m]["by_tag"][tag]
            cells.append(f"{fmt(t['supported_answer_rate'], True)} (n={t['n']})")
        by_tag_rows.append(f"| {tag} | " + " | ".join(cells) + " |")
    out += [
        "",
        "Supported answers by tag:",
        "",
        header,
        "|---|" + "---|" * len(modes),
        *by_tag_rows,
    ]
    for chart in ("answers_by_mode.png", "guard_effect.png"):
        if (ROOT / "results" / chart).exists():
            out += ["", f"![{chart[:-4]}](results/{chart})"]
    misses = results.get("misses", [])
    out += ["", f"#### Every miss ({len(misses)})", ""]
    if misses:
        out += ["| Question | Mode | What went wrong |", "|---|---|---|"]
        for mi in misses:
            why = mi["what_went_wrong"].replace("|", "/")
            out.append(f"| {mi['question_id']} | {LABEL.get(mi['mode'], mi['mode'])} | {why} |")
    else:
        out.append("None.")
    out.append("")
    return out


def main() -> None:
    results = read_results()
    body = [
        START,
        "",
        "Generated by `scripts/update_readme.py` from [results.json](results.json). Do not edit by "
        "hand.",
        "",
        *retrieval_block(results),
        *answers_block(results),
        END,
    ]
    text = README.read_text("utf-8")
    block = "\n".join(body)
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        raise SystemExit(f"README.md has no {START} ... {END} block")
    README.write_text(text, encoding="utf-8", newline="\n")
    print("README.md results block updated")


if __name__ == "__main__":
    main()
