"""
Run the ProbeAI evaluation suite.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --category simple_factual
    python scripts/evaluate.py --category adversarial --top-k 3
    python scripts/evaluate.py --out eval_results/run_001.json
    python scripts/evaluate.py --agent --category multi_field
    python scripts/evaluate.py --multi-agent --category multi_hop
    python scripts/evaluate.py --connector tmdb --category simple_factual
    python scripts/evaluate.py --scenarios scenarios/generated.json --connector tmdb
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.evaluation.metrics.answer_relevancy import score_answer_relevancy
from src.evaluation.metrics.attribution import score_attribution
from src.evaluation.metrics.context_precision import score_context_precision
from src.evaluation.metrics.context_relevancy import score_context_relevancy
from src.evaluation.metrics.coordination import score_coordination
from src.evaluation.metrics.cost import score_multi_agent_cost
from src.evaluation.metrics.cost_per_query import score_cost_per_query
from src.evaluation.metrics.faithfulness import score_faithfulness
from src.evaluation.metrics.grounding import score_grounding
from src.evaluation.metrics.hallucination import score_hallucination
from src.evaluation.metrics.latency import score_latency
from src.evaluation.metrics.mrr import score_mrr
from src.evaluation.metrics.ndcg import score_ndcg
from src.evaluation.metrics.retrieval import score_retrieval
from src.evaluation.metrics.routing import score_routing
from src.evaluation.metrics.task_success import score_task_success
from src.evaluation.metrics.token_efficiency import score_token_efficiency
from src.evaluation.metrics.tool_selection import score_tool_selection
from src.evaluation.metrics.verifier import score_verifier
from src.evaluation.runner import EvalRun, load_scenarios, run

# metrics that apply to both RAG and agent pipelines
SHARED_METRICS = [
    score_faithfulness,
    score_grounding,
    score_hallucination,
    score_latency,
    score_answer_relevancy,   # does the answer address the question?
    score_context_relevancy,  # are the retrieved/tool docs useful?
    score_token_efficiency,   # input + output tokens consumed
    score_cost_per_query,     # estimated dollar cost per query
]

# RAG-only: retrieval quality and ranking metrics need ranked vector results
RAG_METRICS = [
    score_retrieval,
    score_context_precision,  # Precision@K — what fraction of retrieved docs were relevant?
    score_mrr,                # where does the first relevant doc appear in the ranked list?
    score_ndcg,               # are the most relevant docs ranked highest?
] + SHARED_METRICS

# agent adds task_success and tool_selection on top of shared metrics
AGENT_METRICS = SHARED_METRICS + [
    score_task_success,
    score_tool_selection,
]

# multi-agent adds routing, coordination, cost and verifier metrics. These are
# mechanical and return None scores when there's no multi-agent trace.
MULTI_AGENT_METRICS = AGENT_METRICS + [
    score_routing,
    score_coordination,
    score_multi_agent_cost,
    score_verifier,
    score_attribution,  # must come after task_success and hallucination
]


def print_summary(eval_run: EvalRun, is_agent: bool = False, label: str | None = None) -> None:
    """Print a readable summary table to stdout."""
    results = eval_run.results

    mode = label or ("Agent" if is_agent else "RAG")

    # header
    print(f"\n{'='*160}")
    print(f"  ProbeAI Eval Results ({mode}) — {eval_run.total_scenarios} scenarios, model: {eval_run.model}")
    print(f"  Categories: {', '.join(eval_run.categories_run)}")
    print(f"  Total time: {eval_run.total_latency_ms / 1000:.1f}s "
          f"(avg {eval_run.avg_latency_ms / 1000:.1f}s per scenario)")
    print(f"{'='*160}\n")

    if label == "Multi-agent":
        _print_multi_agent_table(results)
    elif is_agent:
        _print_agent_table(results)
    else:
        _print_rag_table(results)

    # aggregate stats
    print(f"\n  {'— Averages —':^150}")

    def avg(key):
        vals = [r.scores.get(key) for r in results if r.scores.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    if not is_agent:
        avg_recall = avg("recall_at_k")
        avg_ctx_prec = avg("context_precision")
        avg_mrr = avg("mrr")
        avg_ndcg = avg("ndcg")
        print(f"  Recall@K:           {avg_recall:.2f}" if avg_recall is not None else "  Recall@K:           n/a")
        print(f"  Context Precision:  {avg_ctx_prec:.2f}" if avg_ctx_prec is not None else "  Context Precision:  n/a")
        print(f"  MRR:                {avg_mrr:.2f}" if avg_mrr is not None else "  MRR:                n/a")
        print(f"  nDCG:               {avg_ndcg:.2f}" if avg_ndcg is not None else "  nDCG:               n/a")

    avg_faith = avg("faithfulness_score")
    avg_ground = avg("grounding_score")
    avg_halluc = avg("hallucination_score")
    avg_ans_rel = avg("answer_relevancy")
    avg_ctx_rel = avg("context_relevancy")
    latency_passes = sum(
        1 for r in results if r.scores.get("latency_pass") is True
    )

    print(f"  Faithfulness:       {avg_faith:.2f}" if avg_faith is not None else "  Faithfulness:       n/a")
    print(f"  Grounding:          {avg_ground:.2f}" if avg_ground is not None else "  Grounding:          n/a")
    print(f"  Hallucination:      {avg_halluc:.2f}" if avg_halluc is not None else "  Hallucination:      n/a")
    print(f"  Answer Relevancy:   {avg_ans_rel:.2f}" if avg_ans_rel is not None else "  Answer Relevancy:   n/a")
    print(f"  Context Relevancy:  {avg_ctx_rel:.2f}" if avg_ctx_rel is not None else "  Context Relevancy:  n/a")
    print(f"  Latency pass:       {latency_passes}/{len(results)}")

    if label == "Multi-agent":
        route_vals = [r.scores.get("routing_accuracy") for r in results if r.scores.get("routing_accuracy") is not None]
        loops = sum(1 for r in results if r.scores.get("loop_detected") is True)
        revised = sum(1 for r in results if r.scores.get("verifier_action") == "revise")
        calls = [r.scores["llm_call_count"] for r in results if r.scores.get("llm_call_count") is not None]
        print(f"  Routing accuracy:   {sum(route_vals) / len(route_vals):.2f}" if route_vals else "  Routing accuracy:   n/a (no expected_route)")
        print(f"  Loops detected:     {loops}/{len(results)}")
        print(f"  Verifier revisions: {revised}/{len(results)}")
        print(f"  LLM calls (avg):    {sum(calls) / len(calls):.1f}" if calls else "  LLM calls (avg):    n/a")

    if is_agent:
        avg_task = avg("task_success_score")
        avg_tool = avg("tool_selection_score")
        print(f"  Task success:       {avg_task:.2f}" if avg_task is not None else "  Task success:       n/a")
        print(f"  Tool selection:     {avg_tool:.2f}" if avg_tool is not None else "  Tool selection:     n/a")

    # operational totals
    token_vals = [r.scores.get("total_tokens") for r in results if r.scores.get("total_tokens") is not None]
    cost_vals = [r.scores.get("cost_usd") for r in results if r.scores.get("cost_usd") is not None]

    if token_vals:
        total_tokens = sum(token_vals)
        avg_tokens = total_tokens / len(token_vals)
        print(f"\n  Tokens (total):     {total_tokens:,}")
        print(f"  Tokens (avg/query): {avg_tokens:,.0f}")
    if cost_vals:
        total_cost = sum(cost_vals)
        avg_cost = total_cost / len(cost_vals)
        print(f"  Cost (total):       ${total_cost:.4f}")
        print(f"  Cost (avg/query):   ${avg_cost:.4f}")

    print()


def _fmt_score(val) -> str:
    """Format a 0–1 score or None for table display."""
    if val is None:
        return "   n/a"
    return f"  {val:.2f}"


def _fmt_tokens(val) -> str:
    """Format a token count integer for table display."""
    if val is None:
        return "    n/a"
    return f"  {val:>5}"


def _fmt_cost(val) -> str:
    """Format a dollar cost for table display."""
    if val is None:
        return "     n/a"
    return f" ${val:.4f}"


def _print_rag_table(results: list) -> None:
    """RAG summary table — 14 metrics: retrieval ranking, generation quality, latency, and ops."""
    header = (
        f"  {'ID':<8} {'Category':<17} {'Recall':>7} {'CtxPre':>7} {'MRR':>7} {'nDCG':>7} "
        f"{'Faith':>7} {'Ground':>7} {'Halluc':>7} "
        f"{'AnsRel':>7} {'CtxRel':>7} {'Tokens':>7} {'Cost$':>8} {'Latency':>10} {'Pass':>5}"
    )
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for r in results:
        s = r.scores
        lat_str = f"{s.get('latency_ms', r.latency_ms):>7.0f}ms"
        lat_pass = s.get("latency_pass")
        pass_str = " ✓" if lat_pass else " ✗" if lat_pass is not None else " n/a"

        print(
            f"  {r.scenario.id:<8} {r.scenario.category:<17}"
            f"{_fmt_score(s.get('recall_at_k'))}"
            f"{_fmt_score(s.get('context_precision'))}"
            f"{_fmt_score(s.get('mrr'))}"
            f"{_fmt_score(s.get('ndcg'))}"
            f"{_fmt_score(s.get('faithfulness_score'))}"
            f"{_fmt_score(s.get('grounding_score'))}"
            f"{_fmt_score(s.get('hallucination_score'))}"
            f"{_fmt_score(s.get('answer_relevancy'))}"
            f"{_fmt_score(s.get('context_relevancy'))}"
            f"{_fmt_tokens(s.get('total_tokens'))}"
            f"{_fmt_cost(s.get('cost_usd'))}"
            f"  {lat_str} {pass_str}"
        )


def _print_agent_table(results: list) -> None:
    """Agent summary table — task/tool metrics + shared quality metrics + ops."""
    header = (
        f"  {'ID':<8} {'Category':<17} {'Task':>6} {'Tools':>6} "
        f"{'Faith':>7} {'Ground':>7} {'Halluc':>7} "
        f"{'AnsRel':>7} {'Tokens':>7} {'Cost$':>8} {'Calls':>6} {'Latency':>10}"
    )
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for r in results:
        s = r.scores
        num_calls = s.get("tool_selection_num_calls", 0)
        lat_str = f"{s.get('latency_ms', r.latency_ms):>7.0f}ms"

        print(
            f"  {r.scenario.id:<8} {r.scenario.category:<17}"
            f"{_fmt_score(s.get('task_success_score'))}"
            f"{_fmt_score(s.get('tool_selection_score'))}"
            f"{_fmt_score(s.get('faithfulness_score'))}"
            f"{_fmt_score(s.get('grounding_score'))}"
            f"{_fmt_score(s.get('hallucination_score'))}"
            f"{_fmt_score(s.get('answer_relevancy'))}"
            f"{_fmt_tokens(s.get('total_tokens'))}"
            f"{_fmt_cost(s.get('cost_usd'))}"
            f"  {num_calls:>4}  {lat_str}"
        )


def _print_multi_agent_table(results: list) -> None:
    """Multi-agent table: task/routing/coordination plus shared quality and cost."""
    header = (
        f"  {'ID':<8} {'Category':<17} {'Task':>6} {'Route':>6} {'Faith':>7} {'Halluc':>7} "
        f"{'AnsRel':>7} {'LLM':>4} {'Hand':>5} {'Verif':>7} {'Tokens':>7} {'Cost$':>8} {'Latency':>10}"
    )
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for r in results:
        s = r.scores
        lat_str = f"{s.get('latency_ms', r.latency_ms):>7.0f}ms"
        loop = "*" if s.get("loop_detected") else " "
        print(
            f"  {r.scenario.id:<8} {r.scenario.category:<17}"
            f"{_fmt_score(s.get('task_success_score'))}"
            f"{_fmt_score(s.get('routing_accuracy'))}"
            f"{_fmt_score(s.get('faithfulness_score'))}"
            f"{_fmt_score(s.get('hallucination_score'))}"
            f"{_fmt_score(s.get('answer_relevancy'))}"
            f"  {s.get('llm_call_count') if s.get('llm_call_count') is not None else '-':>3}"
            f"  {s.get('handoff_count') if s.get('handoff_count') is not None else '-':>3}{loop}"
            f"  {(s.get('verifier_action') or '-'):>6}"
            f"{_fmt_tokens(s.get('total_tokens'))}"
            f"{_fmt_cost(s.get('cost_usd'))}"
            f"  {lat_str}"
        )
    print("\n  (* = loop detected)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ProbeAI evaluation suite")
    parser.add_argument("--category", type=str, default=None,
                        help="filter to a specific scenario category")
    parser.add_argument("--top-k", type=int, default=5,
                        help="number of chunks to retrieve per query (RAG mode)")
    parser.add_argument("--agent", action="store_true",
                        help="run scenarios through the LangGraph agent instead of RAG")
    parser.add_argument("--multi-agent", dest="multi_agent", action="store_true",
                        help="run scenarios through the supervisor + specialists system")
    parser.add_argument("--connector", type=str, default=None,
                        help="use a ProbeConnector instead of the built-in pipeline (e.g. 'tmdb')")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="path to a custom scenarios JSON file (default: scenarios/library.json)")
    parser.add_argument("--out", type=str, default=None,
                        help="path to write full JSON results")
    args = parser.parse_args()

    scenario_path = Path(args.scenarios) if args.scenarios else None
    scenarios = load_scenarios(path=scenario_path, category=args.category) if scenario_path else load_scenarios(category=args.category)

    # resolve the connector if requested
    connector = None
    if args.connector:
        from src.connectors import get_connector
        connector = get_connector(args.connector)
        logger.info(f"using connector: {args.connector}")

    if args.agent and args.multi_agent:
        parser.error("choose one of --agent or --multi-agent")

    if args.multi_agent:
        metrics = MULTI_AGENT_METRICS
    elif args.agent:
        metrics = AGENT_METRICS
    else:
        metrics = RAG_METRICS

    eval_run = run(
        scenarios=scenarios,
        metrics=metrics,
        top_k=args.top_k,
        use_agent=args.agent,
        use_multi_agent=args.multi_agent,
        connector=connector,
    )

    print_summary(eval_run, is_agent=args.agent or args.multi_agent,
                  label="Multi-agent" if args.multi_agent else None)

    # write JSON results if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(eval_run.to_json())
        logger.info(f"results written to {out_path}")


if __name__ == "__main__":
    main()
