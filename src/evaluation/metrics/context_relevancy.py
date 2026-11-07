"""
Context Relevancy: are the retrieved documents actually useful for
answering this specific question?

This is different from Context Precision:
  - Context Precision is mechanical — checks retrieved IDs against ground truth
  - Context Relevancy uses an LLM judge — catches cases where the doc is
    technically the "right" document by ID but its content doesn't help with
    this particular phrasing of the question, or where a doc is topically
    adjacent but missing the specific facts needed

The judge reads the question and all retrieved document chunks, then rates
how useful the documents are as a whole for answering that question.

Works for both RAG and agent pipelines. In agent mode, context_texts contains
tool call outputs rather than vector-retrieved chunks — the judge adapts naturally
since it's just assessing whether the provided context is useful.
"""

import json
from typing import Any

import anthropic
from loguru import logger

from src.config import settings
from src.evaluation.runner import Scenario
from src.generator.prompts import CONTEXT_RELEVANCY_JUDGE_SYSTEM
from src.rag.pipeline import PipelineResult


def _build_prompt(question: str, context_texts: list[str]) -> str:
    """Format the question + retrieved docs for the judge."""
    if not context_texts:
        return f"""Question: {question}

Retrieved documents: (none — no documents were retrieved for this query)

Rate how relevant the retrieved documents are for answering the question."""

    context_block = "\n\n".join(
        f"[Document {i}]\n{text}" for i, text in enumerate(context_texts, 1)
    )

    return f"""Question: {question}

Retrieved documents:
{context_block}

Rate how relevant these documents are for answering the question."""


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
        logger.error(f"[{scenario_id}] could not parse context relevancy judge response")
        return None


def score_context_relevancy(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    LLM-as-judge context relevancy scorer.

    Rates whether the retrieved context (chunks or tool outputs) is
    actually useful for answering this specific question. This catches
    retrieval failures that ground-truth ID checks miss — like when the
    right movie is retrieved but the question asks about a field that's
    buried or absent in that chunk.

    Returns:
      context_relevancy: 0.0-1.0 (1.0 = all docs directly useful)
      context_relevancy_reasoning: judge's one-sentence explanation
      judge_model: which model did the judging
    """
    if not settings.anthropic_api_key:
        return {"context_relevancy": None, "error": "no API key"}

    judge_model = settings.eval_judge_model
    prompt = _build_prompt(question=scenario.question, context_texts=result.context_texts)

    logger.debug(f"[{scenario.id}] calling {judge_model} for context relevancy judging")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=judge_model,
        max_tokens=512,
        system=CONTEXT_RELEVANCY_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    parsed = _parse_response(response.content[0].text, scenario.id)
    if parsed is None:
        return {
            "context_relevancy": None,
            "error": "judge returned unparseable response",
            "raw_response": response.content[0].text,
        }

    score = parsed.get("context_relevancy_score")
    reasoning = parsed.get("reasoning", "")

    if score is None:
        logger.warning(f"[{scenario.id}] judge response missing context_relevancy_score")
        return {"context_relevancy": None, "error": "missing score field"}

    score = round(float(score), 4)
    logger.info(f"[{scenario.id}] context_relevancy={score}")

    return {
        "context_relevancy": score,
        "context_relevancy_reasoning": reasoning,
        "judge_model": judge_model,
    }
