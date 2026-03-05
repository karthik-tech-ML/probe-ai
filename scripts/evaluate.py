"""
Run the ProbeAI evaluation suite.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --category simple_factual
    python scripts/evaluate.py --category adversarial --top-k 3
    python scripts/evaluate.py --out eval_results/run_001.json
    python scripts/evaluate.py --agent --category multi_field
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.evaluation.metrics.faithfulness import score_faithfulness
from src.evaluation.metrics.grounding import score_grounding
from src.evaluation.metrics.hallucination import score_hallucination
from src.evaluation.metrics.latency import score_latency
from src.evaluation.metrics.retrieval import score_retrieval
from src.evaluation.metrics.task_success import score_task_success
from src.evaluation.metrics.tool_selection import score_tool_selection
from src.evaluation.runner import EvalRun, load_scenarios, run

# metrics that apply to both RAG and agent pipelines
SHARED_METRICS = [
    score_faithfulness,
    score_grounding,
    score_hallucination,
    score_latency,
]

# RAG-only: retrieval quality only makes sense with vector search
RAG_METRICS = [
    score_retrieval,
] + SHARED_METRICS

# agent adds task_success and tool_selection on top of shared metrics
AGENT_METRICS = SHARED_METRICS + [
    score_task_success,
    score_tool_selection,
]


def print_summary(eval_run: EvalRun, is_agent: bool = False) -> None:
    """Print a readable summary table to stdout."""
    results = eval_run.results

    mode = "Agent" if is_agent else "RAG"

    # header
    print(f"\n{'='*100}")
    print(f"  ProbeAI Eval Results ({mode}) — {eval_run.total_scenarios} scenarios, model: {eval_run.model}")
    print(f"  Categories: {', '.join(eval_run.categories_run)}")
    print(f"  Total time: {eval_run.total_latency_ms / 1000:.1f}s "
          f"(avg {eval_run.avg_latency_ms / 1000:.1f}s per scenario)")
    print(f"{'='*100}\n")

    if is_agent:
        _print_agent_table(results)
    else:
        _print_rag_table(results)

    # aggregate stats
    print(f"\n  {'— Averages —':^90}")

    def avg(key):
        vals = [r.scores.get(key) for r in results if r.scores.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    if not is_agent:
        avg_recall = avg("recall_at_k")
        print(f"  Recall@K:         {avg_recall:.2f}" if avg_recall is not None else "  Recall@K:         n/a")

    avg_faith = avg("faithfulness_score")
    avg_ground = avg("grounding_score")
    avg_halluc = avg("hallucination_score")
    latency_passes = sum(
        1 for r in results if r.scores.get("latency_pass") is True
    )

    print(f"  Faithfulness:     {avg_faith:.2f}" if avg_faith is not None else "  Faithfulness:     n/a")
    print(f"  Grounding:        {avg_ground:.2f}" if avg_ground is not None else "  Grounding:        n/a")
    print(f"  Hallucination:    {avg_halluc:.2f}" if avg_halluc is not None else "  Hallucination:    n/a")
    print(f"  Latency pass:     {latency_passes}/{len(results)}")

    if is_agent:
        avg_task = avg("task_success_score")
        avg_tool = avg("tool_selection_score")
        print(f"  Task success:     {avg_task:.2f}" if avg_task is not None else "  Task success:     n/a")
        print(f"  Tool selection:   {avg_tool:.2f}" if avg_tool is not None else "  Tool selection:   n/a")

    print()


def _print_rag_table(results: list) -> None:
    """The original RAG summary table."""
    header = (
        f"  {'ID':<8} {'Category':<17} {'Recall':>7} {'Faith':>7} "
        f"{'Ground':>7} {'Halluc':>7} {'Latency':>10} {'Pass':>5}"
    )
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for r in results:
        s = r.scores
        recall = s.get("recall_at_k")
        faith = s.get("faithfulness_score")
        ground = s.get("grounding_score")
        halluc = s.get("hallucination_score")
        lat_ms = s.get("latency_ms", r.latency_ms)
        lat_pass = s.get("latency_pass")

        def fmt(val, invert=False):
            if val is None:
                return "   n/a"
            return f"  {val:.2f}"

        lat_str = f"{lat_ms:>7.0f}ms"
        pass_str = " ✓" if lat_pass else " ✗" if lat_pass is not None else " n/a"

        print(
            f"  {r.scenario.id:<8} {r.scenario.category:<17}"
            f"{fmt(recall)} {fmt(faith)} {fmt(ground)} {fmt(halluc)}"
            f"  {lat_str} {pass_str}"
        )


def _print_agent_table(results: list) -> None:
    """Agent summary table — swaps retrieval for task_success + tool_selection."""
    header = (
        f"  {'ID':<8} {'Category':<17} {'Task':>6} {'Tools':>6} "
        f"{'Faith':>7} {'Ground':>7} {'Halluc':>7} {'Calls':>6} {'Latency':>10}"
    )
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for r in results:
        s = r.scores
        task = s.get("task_success_score")
        tool = s.get("tool_selection_score")
        faith = s.get("faithfulness_score")
        ground = s.get("grounding_score")
        halluc = s.get("hallucination_score")
        num_calls = s.get("tool_selection_num_calls", 0)
        lat_ms = s.get("latency_ms", r.latency_ms)

        def fmt(val):
            if val is None:
                return "   n/a"
            return f"  {val:.2f}"

        lat_str = f"{lat_ms:>7.0f}ms"

        print(
            f"  {r.scenario.id:<8} {r.scenario.category:<17}"
            f"{fmt(task)} {fmt(tool)} {fmt(faith)} {fmt(ground)} {fmt(halluc)}"
            f"  {num_calls:>4}  {lat_str}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ProbeAI evaluation suite")
    parser.add_argument("--category", type=str, default=None,
                        help="filter to a specific scenario category")
    parser.add_argument("--top-k", type=int, default=5,
                        help="number of chunks to retrieve per query (RAG mode)")
    parser.add_argument("--agent", action="store_true",
                        help="run scenarios through the LangGraph agent instead of RAG")
    parser.add_argument("--out", type=str, default=None,
                        help="path to write full JSON results")
    args = parser.parse_args()

    scenarios = load_scenarios(category=args.category)

    if args.agent:
        metrics = AGENT_METRICS
    else:
        metrics = RAG_METRICS

    eval_run = run(
        scenarios=scenarios,
        metrics=metrics,
        top_k=args.top_k,
        use_agent=args.agent,
    )

    print_summary(eval_run, is_agent=args.agent)

    # write JSON results if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(eval_run.to_json())
        logger.info(f"results written to {out_path}")


if __name__ == "__main__":
    main()
