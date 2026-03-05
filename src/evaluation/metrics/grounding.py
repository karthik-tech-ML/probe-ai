"""
Grounding metric: can each factual claim in the answer be traced
to a *specific* source?

Different from faithfulness — faithfulness asks "is this supported
by the context as a whole?" Grounding asks "which exact source
says this?" If a claim is true but can't be pinpointed to a
specific source, it's ungrounded.

Supports two context modes:
  - RAG: trace claims to [Source N] chunk blocks
  - Agent: trace claims to [Tool Call N] results

Detection is automatic via the _agent_trace attribute on PipelineResult.
"""

import json
from typing import Any

import anthropic
from loguru import logger

from src.config import settings
from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult

# --- RAG mode: original prompt expecting [Source N] blocks ---

_RAG_JUDGE_SYSTEM = """\
You are a grounding evaluator. Your job is to trace each factual claim in an \
AI answer back to a specific numbered source.

For each factual claim in the answer:
1. Identify the claim
2. Check if a specific source contains evidence for it
3. If yes, mark it GROUNDED and cite the source number
4. If no single source contains the evidence, mark it UNGROUNDED

A claim is GROUNDED only if you can point to a specific [Source N] that \
contains the supporting information. Vague or general matches don't count.

If the answer is a refusal (says it can't answer or lacks information), \
treat the entire response as GROUNDED — refusing is correct grounding behavior.

Respond with ONLY valid JSON in this exact format, no other text:
{
  "claims": [
    {
      "claim": "short description of the claim",
      "source_id": <int or null>,
      "verdict": "GROUNDED" or "UNGROUNDED"
    }
  ],
  "grounding_score": <float 0.0 to 1.0 = fraction of grounded claims>
}

If the answer has zero factual claims (pure refusal), return:
{
  "claims": [],
  "grounding_score": 1.0
}
"""

# --- Agent mode: trace claims to tool call results ---

_AGENT_JUDGE_SYSTEM = """\
You are a grounding evaluator. Your job is to trace each factual claim in an \
AI answer back to a specific tool call result.

The assistant used tools (database searches, filters, lookups) to gather \
information. Each [Tool Call N] block shows what tool was called and what \
data it returned. The tool outputs contain structured movie data — titles, \
directors, cast lists, budgets, revenue, ratings, genres, years, etc.

For each factual claim in the answer:
1. Identify the claim
2. Check if a specific tool call's output contains evidence for it
3. If yes, mark it GROUNDED and cite the tool call number
4. If no single tool call result contains the evidence, mark it UNGROUNDED

A claim is GROUNDED if you can point to a specific [Tool Call N] whose \
result data supports the claim. Read the tool outputs as structured data — \
look for matching values in the dicts/lists returned by each tool call.

If the answer is a refusal (says it can't answer or lacks information), \
treat the entire response as GROUNDED — refusing is correct grounding behavior.

Respond with ONLY valid JSON in this exact format, no other text:
{
  "claims": [
    {
      "claim": "short description of the claim",
      "source_id": <int tool call number, or null>,
      "verdict": "GROUNDED" or "UNGROUNDED"
    }
  ],
  "grounding_score": <float 0.0 to 1.0 = fraction of grounded claims>
}

If the answer has zero factual claims (pure refusal), return:
{
  "claims": [],
  "grounding_score": 1.0
}
"""


def _is_agent_result(result: PipelineResult) -> bool:
    """Check if this result came from the agent pipeline."""
    return hasattr(result, "_agent_trace") and result._agent_trace is not None


def _build_rag_prompt(question: str, answer: str, context_texts: list[str]) -> str:
    """Format context as numbered [Source N] blocks for RAG mode."""
    context_block = "\n\n".join(
        f"[Source {i}]\n{text}" for i, text in enumerate(context_texts, 1)
    )

    return f"""Context:
{context_block}

Question: {question}

Answer to evaluate:
{answer}

For each factual claim in the answer, identify which specific source (by number) supports it. \
If a claim cannot be traced to any single source, mark it UNGROUNDED."""


def _build_agent_prompt(question: str, answer: str, result: PipelineResult) -> str:
    """
    Format context as labeled tool call blocks for agent mode.

    Shows tool name, input params, and output data so the judge
    can trace each claim to the specific tool call that provided it.
    """
    trace = result._agent_trace
    blocks = []
    for i, tc in enumerate(trace.tool_calls, 1):
        output_str = str(tc.tool_output) if tc.tool_output else "(no output)"
        # cap individual tool outputs so we don't blow the judge context
        if len(output_str) > 4000:
            output_str = output_str[:4000] + "\n... (truncated)"
        blocks.append(
            f"[Tool Call {i}] {tc.tool_name}({json.dumps(tc.tool_input)})\n"
            f"Result:\n{output_str}"
        )

    tool_context = "\n\n".join(blocks)

    return f"""Tool results the assistant had access to:
{tool_context}

Question: {question}

Answer to evaluate:
{answer}

For each factual claim in the answer, identify which specific tool call (by number) supports it. \
If a claim cannot be traced to any single tool call's output, mark it UNGROUNDED."""


def _parse_response(raw: str, scenario_id: str) -> dict | None:
    """Try to parse the judge JSON, handling markdown fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(f"[{scenario_id}] could not parse grounding judge response")
        return None


def score_grounding(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    LLM-as-judge grounding scorer.

    For each claim in the answer, the judge tries to point to the
    specific source (RAG chunk or tool call) that backs it up.
    Automatically picks the right prompt format based on whether
    the result came from RAG or the agent pipeline.

    Returns:
      grounding_score: 0.0-1.0 (fraction of grounded claims)
      grounding_claims: list of {claim, source_id, verdict}
      judge_model: which model did the judging
    """
    if not settings.anthropic_api_key:
        return {"grounding_score": None, "grounding_claims": [], "error": "no API key"}

    judge_model = settings.eval_judge_model
    agent_mode = _is_agent_result(result)

    if agent_mode:
        system_prompt = _AGENT_JUDGE_SYSTEM
        prompt = _build_agent_prompt(
            question=scenario.question,
            answer=result.answer,
            result=result,
        )
    else:
        system_prompt = _RAG_JUDGE_SYSTEM
        prompt = _build_rag_prompt(
            question=scenario.question,
            answer=result.answer,
            context_texts=result.context_texts,
        )

    mode_label = "agent" if agent_mode else "RAG"
    logger.debug(f"[{scenario.id}] calling {judge_model} for grounding judging ({mode_label} mode)")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=judge_model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    parsed = _parse_response(response.content[0].text, scenario.id)
    if parsed is None:
        return {
            "grounding_score": None,
            "grounding_claims": [],
            "error": "judge returned unparseable response",
            "raw_response": response.content[0].text,
        }

    claims = parsed.get("claims", [])

    # recompute score from verdicts — don't trust the model's arithmetic
    if claims:
        grounded = sum(1 for c in claims if c.get("verdict") == "GROUNDED")
        score = round(grounded / len(claims), 4)
    else:
        score = 1.0

    logger.info(f"[{scenario.id}] grounding={score} ({len(claims)} claims)")

    return {
        "grounding_score": score,
        "grounding_claims": claims,
        "judge_model": judge_model,
    }
