"""
Error attribution: when a multi-agent run gets the answer wrong, which agent
is to blame and why? (LLM-as-judge.)

Runs only for wrong answers, judged from the scores already computed for the
run: task_success_score < 1.0, or hallucination_score > HALLUCINATION_GATE. Put this scorer
after those two in the metric list. For right answers it returns None fields
and makes no API call.

Judge results are cached by scenario ID + a hash of the trace, expected
answer and judge model, so re-running an eval doesn't pay for the same
verdict twice. Delete results/.cache/attribution.json to force fresh calls.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

from src.config import settings
from src.evaluation.runner import Scenario
from src.generator.prompts import ATTRIBUTION_JUDGE_SYSTEM
from src.multi_agent.trace import MultiAgentTrace
from src.multi_agent.utils import parse_json_object
from src.rag.pipeline import PipelineResult

# A hallucination score at or below this alone doesn't make an answer "wrong":
# the judge flags the odd unsupported detail even on answers task_success marks
# fully correct. Below the gate, only task_success < 1.0 triggers attribution.
HALLUCINATION_GATE = 0.2

CULPRITS = (
    "supervisor", "retrieval_agent", "structured_agent",
    "analyst_agent", "verifier_agent", "data",
)
FAILURE_TYPES = (
    "wrong_route", "missing_evidence", "bad_tool_args",
    "reasoning_error", "verifier_missed", "other",
)

_CACHE_PATH = Path(__file__).resolve().parents[3] / "results" / ".cache" / "attribution.json"
_MAX_STEP_CHARS = 1500


def _none(reason: str) -> dict[str, Any]:
    return {
        "attribution_culprit_agent": None,
        "attribution_failure_type": None,
        "attribution_reason": reason,
        "attribution_error_propagated": None,
    }


def answer_is_wrong(scores: dict[str, Any]) -> bool | None:
    """True/False from task_success and hallucination; None if neither is available."""
    task = scores.get("task_success_score")
    halluc = scores.get("hallucination_score")
    has_task = isinstance(task, (int, float))
    has_halluc = isinstance(halluc, (int, float))
    if not has_task and not has_halluc:
        return None
    return (has_task and task < 1.0) or (has_halluc and halluc > HALLUCINATION_GATE)


def trace_hash(scenario: Scenario, trace: MultiAgentTrace) -> str:
    """Stable across reruns: full step content, no timestamps."""
    payload = {
        "expected": scenario.expected_answer,
        "answer": trace.final_answer,
        "judge": settings.eval_judge_model,
        "steps": [
            [s.agent_name, s.step_type, s.target, str(s.input), str(s.output)]
            for s in trace.steps
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _render_trace(trace: MultiAgentTrace) -> str:
    lines = []
    for i, s in enumerate(trace.steps, 1):
        target = f" -> {s.target}" if s.target else ""
        lines.append(
            f"{i}. [{s.agent_name}] {s.step_type}{target}\n"
            f"   input:  {str(s.input)[:_MAX_STEP_CHARS]}\n"
            f"   output: {str(s.output)[:_MAX_STEP_CHARS]}"
        )
    return "\n".join(lines)


def _load_cache() -> dict[str, Any]:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _call_judge(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.eval_judge_model,
        max_tokens=512,
        temperature=0,
        system=ATTRIBUTION_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _validate(parsed: dict[str, Any]) -> dict[str, Any]:
    """Keep only values from the allowed sets; flag anything the judge made up."""
    culprit = parsed.get("culprit_agent")
    failure = parsed.get("failure_type")
    invalid = []
    if culprit not in CULPRITS:
        invalid.append(f"culprit_agent={culprit!r}")
        culprit = None
    if failure not in FAILURE_TYPES:
        invalid.append(f"failure_type={failure!r}")
        failure = "other" if culprit else None
    propagated = parsed.get("error_propagated")
    out = {
        "attribution_culprit_agent": culprit,
        "attribution_failure_type": failure,
        "attribution_reason": str(parsed.get("reason", "")),
        "attribution_error_propagated": propagated if isinstance(propagated, bool) else None,
    }
    if invalid:
        out["attribution_invalid_fields"] = invalid
    return out


def score_attribution(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    trace = getattr(result, "_agent_trace", None)
    if not isinstance(trace, MultiAgentTrace):
        return _none("no multi-agent trace")

    wrong = answer_is_wrong(getattr(result, "_scores", {}))
    if wrong is None:
        return _none("no task_success or hallucination score to decide if the answer is wrong")
    if not wrong:
        return _none("answer not wrong, nothing to attribute")
    if not settings.anthropic_api_key:
        return _none("no API key")

    key = f"{scenario.id}:{trace_hash(scenario, trace)}"
    cache = _load_cache()
    if key in cache:
        logger.info(f"[{scenario.id}] attribution cache hit")
        return {**cache[key], "attribution_cached": True}

    prompt = (
        f"Question: {scenario.question}\n\n"
        f"Expected answer: {scenario.expected_answer}\n\n"
        f"Final answer given: {trace.final_answer}\n\n"
        f"Trace:\n{_render_trace(trace)}"
    )
    logger.debug(f"[{scenario.id}] calling {settings.eval_judge_model} for attribution")
    raw = _call_judge(prompt)
    parsed = parse_json_object(raw)
    if parsed is None:
        logger.error(f"[{scenario.id}] could not parse attribution judge response")
        return _none("unparseable judge response")

    verdict = _validate(parsed)
    logger.info(
        f"[{scenario.id}] attribution: {verdict['attribution_culprit_agent']} / "
        f"{verdict['attribution_failure_type']}"
    )
    # don't cache verdicts with made-up fields, so a rerun gets another try
    if "attribution_invalid_fields" not in verdict:
        cache[key] = verdict
        _save_cache(cache)
    return {**verdict, "attribution_cached": False}
