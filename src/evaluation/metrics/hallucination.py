"""
Hallucination detection: what fraction of claims in the answer
have NO support in the retrieved context?

This is the flip of grounding. Grounding asks "how much is
traceable?" Hallucination asks "how much is fabricated?"

  hallucination_score = 0.0 → no hallucination (good)
  hallucination_score = 1.0 → everything is made up (bad)

Supports RAG and agent context modes — same auto-detection as
faithfulness.py and grounding.py via _agent_trace attribute.

Uses the grounding judge results when available to avoid a
redundant LLM call. Falls back to its own judge call if
grounding hasn't run.
"""

import json
from typing import Any

import anthropic
from loguru import logger

from src.config import settings
from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult

# --- RAG mode ---

_RAG_JUDGE_SYSTEM = """\
You are a hallucination detector. Your job is to find claims in an AI answer \
that are NOT supported by the provided source context.

For each factual claim in the answer:
1. Identify the claim
2. Check if any source contains evidence for it
3. If NO source supports the claim, mark it HALLUCINATED
4. If a source supports it, mark it SUPPORTED

A claim is HALLUCINATED if it contains information that does not appear in \
any of the provided sources — even if the information happens to be true \
in the real world. We only care about what the sources say.

If the answer is a refusal (says it can't answer or lacks information), \
that is NOT hallucination. Refusing to answer is correct behavior.

Respond with ONLY valid JSON in this exact format, no other text:
{
  "claims": [
    {
      "claim": "short description of the claim",
      "verdict": "SUPPORTED" or "HALLUCINATED",
      "reason": "brief explanation"
    }
  ],
  "hallucination_score": <float 0.0 to 1.0 = fraction of hallucinated claims>
}

If the answer has zero factual claims (pure refusal), return:
{
  "claims": [],
  "hallucination_score": 0.0
}
"""

# --- Agent mode ---

_AGENT_JUDGE_SYSTEM = """\
You are a hallucination detector. Your job is to find claims in an AI answer \
that are NOT supported by the tool results the assistant received.

The assistant used tools (database searches, filters, lookups) to gather \
information. Each [Tool Call N] block shows what tool was called and what \
data it returned. The tool outputs contain structured movie data — titles, \
directors, cast lists, budgets, revenue, ratings, genres, years, etc.

For each factual claim in the answer:
1. Identify the claim
2. Check if any tool call's output contains evidence for it
3. If NO tool result supports the claim, mark it HALLUCINATED
4. If a tool result supports it, mark it SUPPORTED

A claim is HALLUCINATED if it contains information not present in any tool \
output — even if the information is true in the real world. We only care \
about what the tools returned. Read the tool outputs as structured data — \
check for matching values in the dicts/lists.

If the answer is a refusal (says it can't answer or lacks information), \
that is NOT hallucination. Refusing to answer is correct behavior.

Respond with ONLY valid JSON in this exact format, no other text:
{
  "claims": [
    {
      "claim": "short description of the claim",
      "verdict": "SUPPORTED" or "HALLUCINATED",
      "reason": "brief explanation"
    }
  ],
  "hallucination_score": <float 0.0 to 1.0 = fraction of hallucinated claims>
}

If the answer has zero factual claims (pure refusal), return:
{
  "claims": [],
  "hallucination_score": 0.0
}
"""


def _is_agent_result(result: PipelineResult) -> bool:
    """Check if this result came from the agent pipeline."""
    return hasattr(result, "_agent_trace") and result._agent_trace is not None


def _build_rag_prompt(question: str, answer: str, context_texts: list[str]) -> str:
    context_block = "\n\n".join(
        f"[Source {i}]\n{text}" for i, text in enumerate(context_texts, 1)
    )

    return f"""Context:
{context_block}

Question: {question}

Answer to evaluate:
{answer}

Identify every factual claim in the answer. For each one, determine if it is \
supported by the sources or hallucinated (not present in any source)."""


def _build_agent_prompt(question: str, answer: str, result: PipelineResult) -> str:
    """Format tool call results for the hallucination judge."""
    trace = result._agent_trace
    blocks = []
    for i, tc in enumerate(trace.tool_calls, 1):
        output_str = str(tc.tool_output) if tc.tool_output else "(no output)"
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

Identify every factual claim in the answer. For each one, determine if it is \
supported by the tool results or hallucinated (not present in any tool output)."""


def _parse_response(raw: str, scenario_id: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(f"[{scenario_id}] could not parse hallucination judge response")
        return None


def _derive_from_grounding(scores: dict) -> dict[str, Any] | None:
    """
    If grounding already ran, we can flip its verdicts instead of
    making another LLM call. Returns None if grounding data isn't there.
    """
    grounding_claims = scores.get("grounding_claims")
    grounding_score = scores.get("grounding_score")

    if grounding_claims is None or grounding_score is None:
        return None

    # flip: UNGROUNDED in grounding = HALLUCINATED here
    hallucination_claims = []
    for c in grounding_claims:
        verdict = "HALLUCINATED" if c.get("verdict") == "UNGROUNDED" else "SUPPORTED"
        hallucination_claims.append({
            "claim": c["claim"],
            "verdict": verdict,
            "reason": f"derived from grounding verdict: {c.get('verdict')}",
        })

    if hallucination_claims:
        hallucinated = sum(1 for c in hallucination_claims if c["verdict"] == "HALLUCINATED")
        score = round(hallucinated / len(hallucination_claims), 4)
    else:
        score = 0.0

    return {
        "hallucination_score": score,
        "hallucination_claims": hallucination_claims,
        "derived_from": "grounding",
    }


def score_hallucination(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    Hallucination scorer: fraction of claims with no source support.

    Automatically uses agent-aware prompts when the result came
    from the agent pipeline. Tries to reuse grounding results
    first (saves an API call). Falls back to its own LLM judge
    if grounding hasn't run.

    Returns:
      hallucination_score: 0.0-1.0 (0 = clean, 1 = all fabricated)
      hallucination_claims: list of {claim, verdict, reason}
    """
    if not settings.anthropic_api_key:
        return {"hallucination_score": None, "hallucination_claims": [], "error": "no API key"}

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
    logger.debug(f"[{scenario.id}] calling {judge_model} for hallucination detection ({mode_label} mode)")

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
            "hallucination_score": None,
            "hallucination_claims": [],
            "error": "judge returned unparseable response",
            "raw_response": response.content[0].text,
        }

    claims = parsed.get("claims", [])

    # recompute — don't trust the model's math
    if claims:
        hallucinated = sum(1 for c in claims if c.get("verdict") == "HALLUCINATED")
        score = round(hallucinated / len(claims), 4)
    else:
        score = 0.0

    logger.info(f"[{scenario.id}] hallucination={score} ({len(claims)} claims)")

    return {
        "hallucination_score": score,
        "hallucination_claims": claims,
        "judge_model": judge_model,
    }
