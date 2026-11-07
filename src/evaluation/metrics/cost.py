"""
Multi-agent cost: what did the team spend to get this answer?

Mechanical, no LLM call. Complements cost_per_query (dollars) with the
numbers that explain it: LLM calls, tokens, latency, and tokens per
*correct* answer.

tokens_per_correct_answer is total_tokens when this run's task_success is
1.0, else None. Averaged over a set of runs, "sum of tokens / number
correct" is the number to compare modes on, so the report layer computes
that aggregate from total_tokens and task_success_score; this field is the
per-scenario view of it.
"""

from typing import Any

from src.evaluation.runner import Scenario
from src.multi_agent.trace import MultiAgentTrace
from src.rag.pipeline import PipelineResult


def score_multi_agent_cost(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    trace = getattr(result, "_agent_trace", None)
    if not isinstance(trace, MultiAgentTrace):
        return {
            "llm_call_count": None,
            "ma_total_tokens": None,
            "ma_latency_ms": None,
            "tokens_per_correct_answer": None,
            "cost_reason": "no multi-agent trace",
        }

    total_tokens = trace.total_input_tokens + trace.total_output_tokens
    task_success = getattr(result, "_scores", {}).get("task_success_score")

    return {
        "llm_call_count": trace.llm_call_count,
        "ma_total_tokens": total_tokens,
        "ma_latency_ms": round(result.latency_ms, 1),
        # None when we can't tell (no task_success) or the answer was not fully correct
        "tokens_per_correct_answer": (
            total_tokens if isinstance(task_success, (int, float)) and task_success >= 1.0 else None
        ),
    }
