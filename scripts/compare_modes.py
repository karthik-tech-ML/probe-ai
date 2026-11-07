"""
Three-way comparison: RAG vs single agent vs multi-agent on the same scenarios.

Runs the hard categories (multi_field, aggregation, multi_hop, adversarial;
18 scenarios) through each mode and prints one table per mode, then a
side-by-side summary and a count of multi-agent failures by culprit agent.

    python scripts/compare_modes.py --limit 3     # cheap trial: 3 scenarios, mixed categories
    python scripts/compare_modes.py               # the full 18-scenario subset
    python scripts/compare_modes.py --out results/mode_comparison.json

Cost: multi-agent makes several LLM calls per question, and every mode is
judged (task_success, faithfulness, hallucination). The full run is roughly
18 x 3 questions plus about 4 judge calls each.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.evaluation.metrics.attribution import score_attribution
from src.evaluation.metrics.coordination import score_coordination
from src.evaluation.metrics.cost import score_multi_agent_cost
from src.evaluation.metrics.cost_per_query import score_cost_per_query
from src.evaluation.metrics.faithfulness import score_faithfulness
from src.evaluation.metrics.hallucination import score_hallucination
from src.evaluation.metrics.latency import score_latency
from src.evaluation.metrics.routing import score_routing
from src.evaluation.metrics.task_success import score_task_success
from src.evaluation.metrics.token_efficiency import score_token_efficiency
from src.evaluation.metrics.tool_selection import score_tool_selection
from src.evaluation.metrics.verifier import score_verifier
from src.evaluation.runner import EvalRun, load_scenarios, run

CATEGORIES = ["multi_field", "aggregation", "multi_hop", "adversarial"]
MODES = ["rag", "agent", "multi_agent"]
MODE_LABEL = {"rag": "RAG", "agent": "Single agent", "multi_agent": "Multi-agent"}


def score_llm_calls(scenario, result) -> dict[str, Any]:
    """LLM calls spent answering (judges excluded). RAG is one generation call."""
    trace = getattr(result, "_agent_trace", None)
    if trace is None:
        return {"answer_llm_calls": 1}
    return {"answer_llm_calls": getattr(trace, "llm_calls", None)}


# task_success + hallucination come before attribution, which reads them
_COMMON = [score_task_success, score_faithfulness, score_hallucination, score_latency,
           score_token_efficiency, score_cost_per_query, score_llm_calls]
METRICS = {
    "rag": _COMMON,
    "agent": _COMMON + [score_tool_selection],
    "multi_agent": _COMMON + [score_tool_selection, score_routing, score_coordination,
                              score_multi_agent_cost, score_verifier, score_attribution],
}


def select_scenarios(limit: int | None) -> list:
    """The comparison subset; with --limit, round-robin across categories."""
    by_cat = [load_scenarios(category=c) for c in CATEGORIES]
    ordered = [s for group in zip(*by_cat) for s in group]  # one per category at a time
    ordered += [s for group in by_cat for s in group[min(len(g) for g in by_cat):]]
    return ordered[:limit] if limit else [s for group in by_cat for s in group]


def _num(v: Any) -> float | None:
    """Scorers store {'error': ...} dicts on failure; treat those as missing."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [x for x in (_num(r.get(key)) for r in rows) if x is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(eval_run: EvalRun) -> dict[str, Any]:
    rows = [r.scores for r in eval_run.results]
    correct = [r for r in rows if _num(r.get("task_success_score")) == 1.0]
    tokens = [x for x in (_num(r.get("total_tokens")) for r in rows) if x is not None]
    culprits = Counter(
        f"{r['attribution_culprit_agent']} / {r['attribution_failure_type']}"
        for r in rows if r.get("attribution_culprit_agent")
    )
    return {
        "n": len(rows),
        "accuracy": _avg(rows, "task_success_score"),
        "fully_correct": len(correct),
        "faithfulness": _avg(rows, "faithfulness_score"),
        "hallucination": _avg(rows, "hallucination_score"),
        "routing_accuracy": _avg(rows, "routing_accuracy"),
        "avg_llm_calls": _avg(rows, "answer_llm_calls"),
        "avg_tokens": _avg(rows, "total_tokens"),
        "tokens_per_correct_answer": (sum(tokens) / len(correct)) if correct and tokens else None,
        "avg_latency_ms": _avg(rows, "latency_ms"),
        "avg_cost_usd": _avg(rows, "cost_usd"),
        "total_cost_usd": sum(x for x in (_num(r.get("cost_usd")) for r in rows) if x is not None),
        "loops": sum(1 for r in rows if r.get("loop_detected") is True),
        "verifier_revisions": sum(1 for r in rows if r.get("verifier_action") == "revise"),
        "failures_by_culprit": dict(culprits.most_common()),
    }


def _f(v: float | None, fmt: str = "{:.2f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def print_mode_table(mode: str, eval_run: EvalRun) -> None:
    multi = mode == "multi_agent"
    print(f"\n{'=' * 118}\n  {MODE_LABEL[mode]} — {len(eval_run.results)} scenarios\n{'=' * 118}")
    head = f"  {'ID':<8} {'Category':<13} {'Task':>5} {'Faith':>6} {'Halluc':>7} {'LLM':>4} {'Tokens':>7} {'Cost$':>8} {'Latency':>9}"
    if multi:
        head += f" {'Route':>6} {'Verif':>7}  Culprit"
    print(head)
    for r in eval_run.results:
        s = r.scores
        row = (f"  {r.scenario.id:<8} {r.scenario.category:<13} {_f(_num(s.get('task_success_score'))):>5} "
               f"{_f(_num(s.get('faithfulness_score'))):>6} {_f(_num(s.get('hallucination_score'))):>7} "
               f"{_f(_num(s.get('answer_llm_calls')), '{:.0f}'):>4} {_f(_num(s.get('total_tokens')), '{:.0f}'):>7} "
               f"{_f(_num(s.get('cost_usd')), '${:.4f}'):>8} {_f(_num(s.get('latency_ms')), '{:.0f}ms'):>9}")
        if multi:
            row += (f" {_f(_num(s.get('routing_accuracy'))):>6} {(s.get('verifier_action') or '-'):>7}"
                    f"  {s.get('attribution_culprit_agent') or ''}")
        print(row)


def print_summary(summaries: dict[str, dict]) -> None:
    print(f"\n{'=' * 118}\n  Side by side\n{'=' * 118}")
    print(f"  {'':<28}" + "".join(f"{MODE_LABEL[m]:>16}" for m in summaries))
    rows = [
        ("Accuracy (task_success)", "accuracy", "{:.2f}"), ("Fully correct", "fully_correct", "{:.0f}"),
        ("Faithfulness", "faithfulness", "{:.2f}"), ("Hallucination", "hallucination", "{:.2f}"),
        ("Routing accuracy", "routing_accuracy", "{:.2f}"), ("Avg LLM calls", "avg_llm_calls", "{:.1f}"),
        ("Avg tokens", "avg_tokens", "{:,.0f}"), ("Tokens / correct answer", "tokens_per_correct_answer", "{:,.0f}"),
        ("Avg latency (ms)", "avg_latency_ms", "{:,.0f}"), ("Avg cost / query", "avg_cost_usd", "${:.4f}"),
        ("Total cost", "total_cost_usd", "${:.3f}"),
    ]
    for label, key, fmt in rows:
        print(f"  {label:<28}" + "".join(f"{_f(s.get(key), fmt):>16}" for s in summaries.values()))
    ma = summaries.get("multi_agent")
    if ma:
        print(f"\n  Multi-agent: {ma['loops']} loop(s), {ma['verifier_revisions']} verifier revision(s)")
        print("  Multi-agent failures by culprit / type:",
              ma["failures_by_culprit"] or "none attributed")


def preflight() -> None:
    """One tiny API call, so a dead key or empty balance stops us before the run, not 15 minutes in."""
    import anthropic

    from src.config import settings

    try:
        anthropic.Anthropic(api_key=settings.anthropic_api_key).messages.create(
            model=settings.llm_model, max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    except anthropic.APIError as e:
        sys.exit(f"API preflight failed, not starting the run: {e}")


def check_mode_ran(mode: str, eval_run: EvalRun) -> None:
    """The runner swallows per-scenario exceptions; if most failed, stop instead of comparing blanks."""
    failed = sum(1 for r in eval_run.results if "error" in r.scores and r.scores.get("task_success_score") is None
                 and r.latency_ms == 0)
    if eval_run.results and failed > len(eval_run.results) / 2:
        sys.exit(f"{failed}/{len(eval_run.results)} scenarios failed in mode {mode!r}; "
                 f"aborting. First error: {eval_run.results[0].answer[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG, single agent and multi-agent")
    parser.add_argument("--limit", type=int, default=None, help="only run this many scenarios (mixed categories)")
    parser.add_argument("--modes", default=",".join(MODES), help="comma-separated subset of: " + ",".join(MODES))
    parser.add_argument("--out", default="results/mode_comparison.json")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    bad = [m for m in modes if m not in MODES]
    if bad:
        parser.error(f"unknown mode(s): {bad}")

    preflight()
    scenarios = select_scenarios(args.limit)
    logger.info(f"comparing {modes} on {len(scenarios)} scenarios: {[s.id for s in scenarios]}")

    runs: dict[str, EvalRun] = {}
    for mode in modes:
        logger.info(f"===== mode: {mode} =====")
        runs[mode] = run(scenarios=scenarios, metrics=METRICS[mode],
                         use_agent=(mode == "agent"), use_multi_agent=(mode == "multi_agent"))
        print_mode_table(mode, runs[mode])
        check_mode_ran(mode, runs[mode])

    summaries = {m: summarize(r) for m, r in runs.items()}
    print_summary(summaries)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scenarios": [s.id for s in scenarios],
        "summary": summaries,
        "runs": {m: r.to_dict() for m, r in runs.items()},
    }, indent=2))
    logger.info(f"results written to {out}")


if __name__ == "__main__":
    main()
