"""Draw charts into results/ from results.json (never from typed numbers).

Charts are only drawn for blocks that exist: the retrieval chart needs `retrieval_eval`
(scripts/eval_retrieval.py), the answer charts need `modes` (scripts/measure.py).

Usage: uv run python scripts/plot_results.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import ROOT, read_results  # noqa: E402

OUT = ROOT / "results"
COLORS = {"vector": "#6b7a8f", "hybrid": "#2f6f9f", "hybrid_rerank": "#1f9e74"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _bars(ax, labels, series: dict[str, list[float | None]], ylabel: str, ylim=(0, 1.05)):
    n = len(series)
    width = 0.8 / max(n, 1)
    for i, (name, values) in enumerate(series.items()):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        vals = [v if v is not None else 0 for v in values]
        bars = ax.bar(xs, vals, width, label=name, color=COLORS.get(name, None))
        for b, v in zip(bars, values, strict=True):
            text = "n/m" if v is None else f"{v:.2f}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncols=n, loc="upper center", bbox_to_anchor=(0.5, 1.12))


def plot_retrieval(results: dict) -> list[str]:
    written = []
    for model, block in results.get("retrieval_eval", {}).items():
        modes = [m for m, v in block["modes"].items() if "recall_at_k" in v]
        if not modes:
            continue
        metrics = [
            ("recall@k", lambda v: v["recall_at_k"]),
            ("hit@1", lambda v: v.get("hit_at_1")),
            ("MRR", lambda v: v["mrr"]),
            ("recall@k exact-term", lambda v: v["by_tag"]["exact-term"]["recall_at_k"]),
            ("recall@k paraphrase", lambda v: v["by_tag"]["paraphrase"]["recall_at_k"]),
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        _bars(
            ax,
            [m for m, _ in metrics],
            {mode: [f(block["modes"][mode]) for _, f in metrics] for mode in modes},
            "score",
        )
        ax.set_title(
            f"Retrieval, {block['questions']} answerable questions, k={block['k']}\n"
            f"embeddings: {model} ({block['date']})",
            fontsize=10,
            pad=28,
        )
        safe = model.split("/")[-1]
        path = OUT / f"retrieval_{safe}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)
    return written


def plot_answers(results: dict) -> list[str]:
    modes = list(results.get("modes", {}))
    if not modes:
        return []
    m = results["modes"]
    metrics = [
        ("supported\nanswers", "supported_answer_rate"),
        ("fully\nsupported", "fully_supported_answer_rate"),
        ("valid\ncitations", "valid_citation_rate"),
        ("page\naccuracy", "page_accuracy"),
        ("abstention\nrecall", "abstention_recall"),
        ("false\nabstention", "false_abstention"),
        ("escalated\n(needs_human)", "escalation_rate"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _bars(
        ax,
        [label for label, _ in metrics],
        {mode: [m[mode].get(key) for _, key in metrics] for mode in modes},
        "rate",
    )
    ax.set_title(
        f"Answers, {results['question_count']} questions x {results['runs']} runs, "
        f"{results['claude_model']} ({results['date']})",
        fontsize=10,
        pad=28,
    )
    fig.tight_layout()
    fig.savefig(OUT / "answers_by_mode.png", dpi=150)
    plt.close(fig)

    base = results.get("baseline_no_guard", {})
    written = ["answers_by_mode.png"]
    if base.get("of_answers"):
        shipped = sum(m[mode]["unsupported_shipped"] for mode in modes)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        labels = ["without the guard\n(first answer as written)", "with the guard\n(shipped)"]
        failing = base.get("answers_failing_evidence_check", base["answers_with_invalid_quote"])
        values = [failing, shipped]
        bars = ax.bar(labels, values, color=["#c0504d", "#1f9e74"], width=0.5)
        for b, v in zip(bars, values, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v), ha="center", va="bottom")
        ax.set_ylabel("answers that fail the evidence check")
        ax.set_title(
            f"What the evidence check buys (baseline: {base['of_answers']} answers)", fontsize=10
        )
        fig.tight_layout()
        fig.savefig(OUT / "guard_effect.png", dpi=150)
        plt.close(fig)
        written.append("guard_effect.png")
    return written


def main() -> None:
    results = read_results()
    if not results:
        raise SystemExit("results.json not found: run scripts/eval_retrieval.py or measure.py")
    OUT.mkdir(exist_ok=True)
    written = plot_retrieval(results) + plot_answers(results)
    print("wrote " + (", ".join(f"results/{w}" for w in written) if written else "nothing"))


if __name__ == "__main__":
    main()
