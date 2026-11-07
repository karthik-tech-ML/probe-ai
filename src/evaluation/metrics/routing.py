"""
Routing accuracy: did the supervisor send the work to the right specialists?

Compares the specialists the supervisor actually routed to against
scenario.expected_route. Mechanical, no LLM call.

- routing_precision: of the specialists used, how many were expected
- routing_recall:    of the expected specialists, how many were used
- routing_accuracy:  F1 of the two (set-level, order ignored)
- routing_exact_order: used route == expected route, repeats collapsed
"""

from typing import Any

from src.evaluation.runner import Scenario
from src.multi_agent.trace import MultiAgentTrace
from src.rag.pipeline import PipelineResult


def collapse_repeats(route: list[str]) -> list[str]:
    """['a', 'a', 'b', 'a'] -> ['a', 'b', 'a']. Back-to-back repeats are one hop."""
    out: list[str] = []
    for name in route:
        if not out or out[-1] != name:
            out.append(name)
    return out


def _none(reason: str) -> dict[str, Any]:
    return {
        "routing_accuracy": None,
        "routing_precision": None,
        "routing_recall": None,
        "routing_exact_order": None,
        "routing_reason": reason,
    }


def score_routing(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    trace = getattr(result, "_agent_trace", None)
    if not isinstance(trace, MultiAgentTrace):
        return _none("no multi-agent trace")
    if not scenario.expected_route:
        return _none("scenario has no expected_route")

    used = collapse_repeats(trace.route_sequence)
    expected = collapse_repeats(scenario.expected_route)
    used_set, expected_set = set(used), set(expected)
    hits = len(used_set & expected_set)

    precision = hits / len(used_set) if used_set else 0.0
    recall = hits / len(expected_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "routing_accuracy": round(f1, 4),
        "routing_precision": round(precision, 4),
        "routing_recall": round(recall, 4),
        "routing_exact_order": used == expected,
        "routing_used": used,
        "routing_expected": expected,
    }
