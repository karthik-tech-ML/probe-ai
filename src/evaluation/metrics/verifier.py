"""
Verifier behaviour: did it approve or ask for a revision, and did that matter?

Mechanical, no LLM call. Only the first verification is meaningful (the
graph allows at most one revise loop and does not re-verify), so
verifier_action is the first verdict.

- verifier_action: 'approve' | 'revise' | None (verifier never ran, e.g. step cap)
- verifier_revision_changed_answer: after a revise, did the final answer
  differ from the draft the verifier rejected? None unless it revised.
"""

import re
from typing import Any

from src.evaluation.runner import Scenario
from src.multi_agent.trace import MultiAgentTrace
from src.rag.pipeline import PipelineResult


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def score_verifier(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    trace = getattr(result, "_agent_trace", None)
    if not isinstance(trace, MultiAgentTrace):
        return {
            "verifier_action": None,
            "verifier_revision_changed_answer": None,
            "verifier_reason": "no multi-agent trace",
        }

    actions = trace.verifier_actions
    if not actions:
        return {
            "verifier_action": None,
            "verifier_revision_changed_answer": None,
            "verifier_reason": "verifier did not run" + (" (step cap hit)" if trace.step_cap_hit else ""),
        }

    action = actions[0]
    changed = None
    if action == "revise":
        rejected_draft = trace.steps_of("verify")[0].input
        changed = _norm(rejected_draft) != _norm(trace.final_answer)

    return {
        "verifier_action": action,
        "verifier_revision_changed_answer": changed,
    }
