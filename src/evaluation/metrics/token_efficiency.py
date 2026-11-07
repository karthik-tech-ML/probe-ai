"""
Token Efficiency: how many tokens did this query consume?

Tracks the raw token counts that are already in the pipeline response —
input tokens (question + context + system prompt) and output tokens
(the generated answer). No LLM call needed; we're just reading what
the API already gave us.

For local inference (Ollama, etc.) where the response doesn't include
token counts, we fall back to a rough character-based estimate using
~4 chars per token. The estimate is flagged so you can distinguish it
from real API counts in reports.

Returns raw counts, not a 0–1 score. These feed into cost tracking and
operational dashboards, not pass/fail evaluation.
"""

from typing import Any

from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult


def _estimate_from_text(scenario: Scenario, result: PipelineResult) -> tuple[int, int]:
    """
    Rough token estimate when real counts aren't available.

    Input estimate covers: context chunks + question text.
    Output estimate covers: the answer text.
    ~4 chars per token is the standard approximation for English text.
    """
    context = "\n".join(result.context_texts) if result.context_texts else ""
    input_chars = len(context) + len(scenario.question)
    output_chars = len(result.answer)

    return max(1, input_chars // 4), max(1, output_chars // 4)


def score_token_efficiency(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    Extract or estimate token counts for this query.

    Reads from result.generation.usage (populated by the Claude API
    response for both RAG and agent paths). Falls back to a character-
    based estimate when token counts aren't available, e.g. when running
    against a local inference server that doesn't report usage.

    Returns:
      input_tokens:  tokens consumed by the prompt
      output_tokens: tokens in the generated answer
      total_tokens:  input + output
      token_count_estimated: True if we fell back to the text estimate
    """
    usage = {}
    if result.generation and hasattr(result.generation, "usage"):
        usage = result.generation.usage or {}

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    estimated = input_tokens is None or output_tokens is None

    if estimated:
        input_est, output_est = _estimate_from_text(scenario, result)
        input_tokens = input_tokens if input_tokens is not None else input_est
        output_tokens = output_tokens if output_tokens is not None else output_est

    total_tokens = input_tokens + output_tokens

    out: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if estimated:
        out["token_count_estimated"] = True

    return out
