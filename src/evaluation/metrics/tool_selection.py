"""
Tool selection metric: was the agent's tool usage reasonable?

Checks three things:
1. Did the agent actually use tools (vs. just guessing from training data)?
2. Did it avoid spamming unnecessary calls?
3. For scenarios with expected_source_ids, did the tools return data
   that covers those movies?

No LLM call needed — this is all mechanical checks on the agent trace.
"""

from typing import Any

from loguru import logger

from src.agent.memory import AgentTrace
from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult


# arbitrary but reasonable: if the agent calls 10+ tools for a single
# question, something went wrong (retry loops, confusion, etc.)
_MAX_REASONABLE_CALLS = 8


def score_tool_selection(
    scenario: Scenario,
    result: PipelineResult,
    trace: AgentTrace | None = None,
) -> dict[str, Any]:
    """
    Score the agent's tool usage quality.

    Takes the scenario + pipeline result (for interface compat with MetricFn)
    AND an optional AgentTrace. When called through the agent eval path,
    trace is attached to the PipelineResult via the _agent_trace attribute.

    Returns a 0-1 composite score plus breakdowns:
    - used_tools: did it call at least one tool?
    - reasonable_count: didn't over-call?
    - coverage: for scenarios with expected_source_ids, what fraction of
      expected movies appeared in tool outputs?
    """
    # grab trace from result if not passed directly — the agent runner
    # stashes it on the PipelineResult for exactly this purpose
    if trace is None:
        trace = getattr(result, "_agent_trace", None)

    if trace is None:
        logger.warning(f"[{scenario.id}] no agent trace — can't score tool selection")
        return {
            "tool_selection_score": None,
            "tool_selection_reason": "no agent trace available",
        }

    num_calls = trace.num_tool_calls
    tools_used = trace.tools_used

    # --- check 1: did the agent use any tools? ---
    used_tools = num_calls > 0
    used_tools_score = 1.0 if used_tools else 0.0

    # --- check 2: reasonable number of calls? ---
    # perfect if <= half the max, degrades linearly, 0 if way over
    if num_calls <= _MAX_REASONABLE_CALLS:
        reasonable_score = 1.0
    else:
        # penalize but don't instantly zero — overshoot by 2x = 0
        overshoot = num_calls - _MAX_REASONABLE_CALLS
        reasonable_score = max(0.0, 1.0 - overshoot / _MAX_REASONABLE_CALLS)

    # --- check 3: did tool results cover the expected movies? ---
    expected_ids = set(scenario.expected_source_ids)
    if expected_ids:
        # scan all tool outputs for movie_id mentions
        found_ids = _extract_movie_ids_from_trace(trace)
        covered = expected_ids & found_ids
        coverage_score = len(covered) / len(expected_ids)
        coverage_detail = {
            "expected": sorted(expected_ids),
            "found_in_tools": sorted(covered),
            "missing": sorted(expected_ids - found_ids),
        }
    else:
        # no expected sources to check (open-ended scenarios) — skip this axis
        coverage_score = None
        coverage_detail = "no expected_source_ids for this scenario"

    # --- composite score ---
    # weight: 30% tool usage + 20% reasonable count + 50% coverage
    # if coverage is N/A, reweight to 60% usage + 40% reasonable
    if coverage_score is not None:
        composite = (
            0.3 * used_tools_score
            + 0.2 * reasonable_score
            + 0.5 * coverage_score
        )
    else:
        composite = (
            0.6 * used_tools_score
            + 0.4 * reasonable_score
        )

    composite = round(composite, 2)

    logger.info(
        f"[{scenario.id}] tool_selection={composite} "
        f"(calls={num_calls}, tools={tools_used}, "
        f"coverage={coverage_score})"
    )

    return {
        "tool_selection_score": composite,
        "tool_selection_used_tools": used_tools,
        "tool_selection_num_calls": num_calls,
        "tool_selection_tools_used": tools_used,
        "tool_selection_reasonable_score": round(reasonable_score, 2),
        "tool_selection_coverage_score": (
            round(coverage_score, 2) if coverage_score is not None else None
        ),
        "tool_selection_coverage_detail": coverage_detail,
    }


def _extract_movie_ids_from_trace(trace: AgentTrace) -> set[int]:
    """
    Pull movie_id values out of tool call outputs.

    Tool outputs are stringified lists of dicts (from our tools.py).
    We look for 'movie_id' keys in the parsed structures, and also
    check for movie_id in the raw string as a fallback.
    """
    import ast
    import re

    found = set()

    for tc in trace.tool_calls:
        output = tc.tool_output
        if output is None:
            continue

        # tool outputs come back as strings from LangGraph
        if isinstance(output, str):
            # try parsing as a Python literal (list of dicts)
            try:
                parsed = ast.literal_eval(output)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "movie_id" in item:
                            found.add(int(item["movie_id"]))
                elif isinstance(parsed, dict) and "movie_id" in parsed:
                    found.add(int(parsed["movie_id"]))
            except (ValueError, SyntaxError):
                pass

            # fallback: regex for movie_id patterns in the raw string
            # catches cases where ast.literal_eval chokes on the format
            for match in re.finditer(r"'movie_id':\s*(\d+)", output):
                found.add(int(match.group(1)))
            for match in re.finditer(r'"movie_id":\s*(\d+)', output):
                found.add(int(match.group(1)))

        elif isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and "movie_id" in item:
                    found.add(int(item["movie_id"]))

        elif isinstance(output, dict) and "movie_id" in output:
            found.add(int(output["movie_id"]))

    return found
