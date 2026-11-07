"""
Coordination: how much hand-wringing did the team do, and did it loop?

Mechanical, no LLM call.

- handoff_count:  handoffs between agents (supervisor -> specialist and back)
- loop_detected:  the supervisor gave the same agent the same task twice,
                  or the step cap was hit
- steps_to_answer: trace steps recorded before the final answer
"""

import re
from typing import Any

from src.evaluation.runner import Scenario
from src.multi_agent.trace import SUPERVISOR, MultiAgentTrace
from src.rag.pipeline import PipelineResult


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def score_coordination(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    trace = getattr(result, "_agent_trace", None)
    if not isinstance(trace, MultiAgentTrace):
        return {
            "handoff_count": None,
            "loop_detected": None,
            "loop_reason": None,
            "steps_to_answer": None,
            "coordination_reason": "no multi-agent trace",
        }

    seen: set[tuple[str, str]] = set()
    repeated = False
    for s in trace.steps_of("handoff"):
        if s.agent_name != SUPERVISOR:
            continue
        key = (str(s.target), _norm(s.input))
        if key in seen:
            repeated = True
        seen.add(key)

    if trace.step_cap_hit:
        reason = "step_cap"
    elif repeated:
        reason = "repeated_call"
    else:
        reason = None

    return {
        "handoff_count": trace.handoff_count,
        "loop_detected": reason is not None,
        "loop_reason": reason,
        "steps_to_answer": len([s for s in trace.steps if s.step_type != "final"]),
        "step_cap_hit": trace.step_cap_hit,
    }
