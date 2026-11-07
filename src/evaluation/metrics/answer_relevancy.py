"""
Answer Relevancy: does the answer actually address what was asked?

This is different from faithfulness. Faithfulness asks: "is the answer
supported by the context?" Answer relevancy asks: "does the answer
respond to the question?"

An answer can be perfectly faithful (every claim is in the context)
but still irrelevant — e.g., it answers a different question, or buries
the actual answer in a wall of unrelated detail.

Works for both RAG and agent pipelines: it only looks at the question
and the final answer, not at the retrieved context.
"""

import json
from typing import Any

import anthropic
from loguru import logger

from src.config import settings
from src.evaluation.runner import Scenario
from src.generator.prompts import ANSWER_RELEVANCY_JUDGE_SYSTEM
from src.rag.pipeline import PipelineResult


def _build_prompt(question: str, answer: str) -> str:
    """Format the question + answer for the judge."""
    return f"""Question: {question}

Answer to evaluate:
{answer}

Rate how well this answer addresses the question."""


def _parse_response(raw: str, scenario_id: str) -> dict | None:
    """Try to parse judge JSON, handling markdown fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(f"[{scenario_id}] could not parse answer relevancy judge response")
        return None


def score_answer_relevancy(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    LLM-as-judge answer relevancy scorer.

    Asks Claude whether the answer actually addresses the question.
    Only looks at question vs. answer — not at the retrieved context.
    This makes it complementary to faithfulness (which looks at context).

    Returns:
      answer_relevancy: 0.0-1.0 (1.0 = directly and completely addresses the question)
      answer_relevancy_reasoning: judge's one-sentence explanation
      judge_model: which model did the judging
    """
    if not settings.anthropic_api_key:
        return {"answer_relevancy": None, "error": "no API key"}

    judge_model = settings.eval_judge_model
    prompt = _build_prompt(question=scenario.question, answer=result.answer)

    logger.debug(f"[{scenario.id}] calling {judge_model} for answer relevancy judging")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=judge_model,
        max_tokens=512,
        system=ANSWER_RELEVANCY_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    parsed = _parse_response(response.content[0].text, scenario.id)
    if parsed is None:
        return {
            "answer_relevancy": None,
            "error": "judge returned unparseable response",
            "raw_response": response.content[0].text,
        }

    score = parsed.get("answer_relevancy_score")
    reasoning = parsed.get("reasoning", "")

    if score is None:
        logger.warning(f"[{scenario.id}] judge response missing answer_relevancy_score")
        return {"answer_relevancy": None, "error": "missing score field"}

    score = round(float(score), 4)
    logger.info(f"[{scenario.id}] answer_relevancy={score}")

    return {
        "answer_relevancy": score,
        "answer_relevancy_reasoning": reasoning,
        "judge_model": judge_model,
    }
