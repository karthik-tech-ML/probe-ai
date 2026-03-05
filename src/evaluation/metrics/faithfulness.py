"""
Faithfulness scoring via LLM-as-judge.

Makes a second Claude call to check whether every claim in the
generated answer is actually supported by the retrieved context.
Returns a 0-1 score plus the individual claim verdicts.

Supports two context modes:
  - RAG: context is numbered [Source N] chunk blocks
  - Agent: context is tool call results (structured dicts, lists, etc.)

The judge prompt adapts automatically based on which mode produced
the result. Detection is based on the _agent_trace attribute that
the agent eval path stashes on the PipelineResult.
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
You are an evaluation judge. Your job is to assess whether an AI assistant's \
answer is faithful to the provided source context.

For each factual claim in the answer, decide if it is SUPPORTED or UNSUPPORTED \
by the context. A claim is SUPPORTED only if the context contains evidence for it. \
If the answer says it lacks information or cannot answer, that counts as SUPPORTED \
(refusing to answer is faithful behavior).

Respond with ONLY valid JSON in this exact format, no other text:
{
  "claims": [
    {"claim": "short description of the claim", "verdict": "SUPPORTED" or "UNSUPPORTED"}
  ],
  "faithfulness_score": <float 0.0 to 1.0 = fraction of supported claims>
}

If the answer has zero factual claims (e.g. it's purely a refusal), return:
{
  "claims": [],
  "faithfulness_score": 1.0
}
"""

# --- Agent mode: prompt adapted for tool output context ---

_AGENT_JUDGE_SYSTEM = """\
You are an evaluation judge. Your job is to assess whether an AI assistant's \
answer is faithful to the tool results it received.

The assistant used tools (database searches, filters, lookups) to gather \
information, then wrote an answer. The tool results are shown below. Each \
[Tool Call N] block shows what tool was called and what data it returned.

For each factual claim in the answer, decide if it is SUPPORTED or UNSUPPORTED \
by the tool results. A claim is SUPPORTED if the data returned by any tool call \
contains evidence for it — look at movie titles, directors, cast lists, budgets, \
revenue figures, ratings, genres, years, and other structured fields in the tool \
outputs. The tool outputs may be Python dicts/lists, so read them as structured data.

If the answer says it lacks information or cannot answer, that counts as SUPPORTED \
(refusing to answer is faithful behavior).

Respond with ONLY valid JSON in this exact format, no other text:
{
  "claims": [
    {"claim": "short description of the claim", "verdict": "SUPPORTED" or "UNSUPPORTED"}
  ],
  "faithfulness_score": <float 0.0 to 1.0 = fraction of supported claims>
}

If the answer has zero factual claims (e.g. it's purely a refusal), return:
{
  "claims": [],
  "faithfulness_score": 1.0
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

Extract each factual claim from the answer and judge whether it is supported by the context."""


def _build_agent_prompt(question: str, answer: str, result: PipelineResult) -> str:
    """
    Format context as labeled tool call blocks for agent mode.

    Uses the actual AgentTrace to show tool name + input + output,
    which gives the judge much better signal than raw context_texts.
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

Extract each factual claim from the answer and judge whether it is supported by the tool results."""


def score_faithfulness(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    LLM-as-judge faithfulness scorer.

    Sends the answer + context to a second Claude call that breaks
    the answer into claims and checks each one. Automatically picks
    the right prompt format based on whether the result came from
    RAG or the agent pipeline.

    Returns:
      faithfulness_score: 0.0-1.0 (fraction of claims supported)
      claims: list of {claim, verdict} dicts
      judge_model: which model did the judging
    """
    if not settings.anthropic_api_key:
        return {"faithfulness_score": None, "claims": [], "error": "no API key"}

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
    logger.debug(f"[{scenario.id}] calling {judge_model} for faithfulness judging ({mode_label} mode)")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=judge_model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"[{scenario.id}] judge returned invalid JSON, attempting repair")
        # sometimes the model wraps JSON in markdown fences
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"[{scenario.id}] could not parse judge response")
            return {
                "faithfulness_score": None,
                "claims": [],
                "error": "judge returned unparseable response",
                "raw_response": raw,
            }

    score = parsed.get("faithfulness_score")
    claims = parsed.get("claims", [])

    # sanity check: recompute score from claims in case the model got it wrong
    if claims:
        supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
        computed_score = round(supported / len(claims), 4)
        if score is not None and abs(score - computed_score) > 0.01:
            logger.debug(
                f"[{scenario.id}] judge self-reported {score}, "
                f"recomputed {computed_score} — using recomputed"
            )
        score = computed_score
    elif score is None:
        score = 1.0

    logger.info(f"[{scenario.id}] faithfulness={score} ({len(claims)} claims)")

    return {
        "faithfulness_score": score,
        "faithfulness_claims": claims,
        "judge_model": judge_model,
    }
