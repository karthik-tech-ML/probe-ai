"""
Cost per Query: estimated dollar cost based on token usage and model pricing.

Uses the same token counts that token_efficiency.py reads — either real
API usage data or the character-based estimate — and multiplies by the
model's per-token price.

Pricing constants (USD per 1M tokens, as of 2025):
  Claude Sonnet:  $3.00 input,  $15.00 output
  Claude Haiku:   $0.25 input,   $1.25 output
  Local (Ollama): $0.00 (free)

Model tier is detected from the model string in the pipeline result.
Anything unrecognized (e.g. a local model name) falls through to $0.00
so cost tracking doesn't blow up on unexpected model identifiers.

Returns raw dollar amounts, not a 0–1 score. These are purely for
reporting — budget tracking, cost-per-accuracy comparisons, etc.
"""

from typing import Any

from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult

# (input_price_per_mtok, output_price_per_mtok) in USD
_PRICING: dict[str, tuple[float, float]] = {
    "sonnet": (3.00, 15.00),
    "haiku":  (0.25,  1.25),
    "opus":   (15.00, 75.00),  # included for completeness
    "local":  (0.00,   0.00),
}


def _detect_tier(model: str) -> str:
    """Map a model name string to a pricing tier."""
    m = model.lower()
    if "haiku" in m:
        return "haiku"
    if "opus" in m:
        return "opus"
    if "sonnet" in m or "claude" in m:
        # covers claude-sonnet-*, claude-3-5-sonnet-*, claude-sonnet-*
        return "sonnet"
    # local inference, ollama, unrecognized identifiers
    return "local"


def _get_tokens(scenario: Scenario, result: PipelineResult) -> tuple[int, int]:
    """
    Read token counts from generation.usage, with character-based fallback.

    Mirrors the same logic as token_efficiency.py so cost_per_query is
    self-contained and doesn't depend on that scorer having run first.
    """
    usage = {}
    if result.generation and hasattr(result.generation, "usage"):
        usage = result.generation.usage or {}

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    if input_tokens is None:
        context = "\n".join(result.context_texts) if result.context_texts else ""
        input_tokens = max(1, (len(context) + len(scenario.question)) // 4)

    if output_tokens is None:
        output_tokens = max(1, len(result.answer) // 4)

    return input_tokens, output_tokens


def score_cost_per_query(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    Estimate query cost from token usage and model pricing.

    Formula: cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    Returns:
      cost_usd: estimated cost in US dollars (e.g. 0.007293)
      cost_model: model string used for pricing lookup
      cost_input_price_per_mtok: price applied to input tokens
      cost_output_price_per_mtok: price applied to output tokens
    """
    model = ""
    if result.generation and hasattr(result.generation, "model"):
        model = result.generation.model or ""

    tier = _detect_tier(model)
    input_price, output_price = _PRICING[tier]

    input_tokens, output_tokens = _get_tokens(scenario, result)

    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    return {
        "cost_usd": round(cost, 6),
        "cost_model": model or "unknown",
        "cost_input_price_per_mtok": input_price,
        "cost_output_price_per_mtok": output_price,
    }
