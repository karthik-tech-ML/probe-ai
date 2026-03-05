"""
Task success metric: did the agent arrive at the correct answer?

Uses LLM-as-judge to compare the agent's answer against the
expected answer from the scenario. More flexible than exact
string matching — "Christopher Nolan" and "The director is
Christopher Nolan" should both score 1.0.
"""

import json
from typing import Any

import anthropic
from loguru import logger

from src.config import settings
from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult

_JUDGE_SYSTEM = """\
You are an evaluation judge. Compare an AI assistant's answer against an \
expected answer for a movie database question.

Score the answer on correctness:
- 1.0: The answer correctly addresses the question and matches the expected \
answer in substance (exact wording not required).
- 0.5: The answer is partially correct — gets some facts right but misses \
key parts of the expected answer, or includes the right info among some errors.
- 0.0: The answer is wrong, doesn't address the question, or fabricates \
information that contradicts the expected answer.

For "no answer" scenarios where the expected answer says the info isn't \
available: score 1.0 if the assistant correctly refuses, 0.0 if it makes \
up an answer.

For adversarial scenarios with false premises: score 1.0 if the assistant \
corrects the false premise using data, 0.5 if it partially corrects, \
0.0 if it accepts the false premise.

Respond with ONLY valid JSON:
{
  "task_success_score": <float 0.0, 0.5, or 1.0>,
  "reason": "brief explanation of the score"
}
"""


def score_task_success(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    LLM-as-judge task success scorer.

    Compares the generated answer against the expected answer,
    accounting for paraphrasing and partial correctness.
    """
    if not settings.anthropic_api_key:
        return {"task_success_score": None, "error": "no API key"}

    judge_model = settings.eval_judge_model

    prompt = f"""Question: {scenario.question}

Expected answer: {scenario.expected_answer}

Actual answer: {result.answer}

Does the actual answer correctly address the question based on the expected answer?"""

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=judge_model,
        max_tokens=256,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"[{scenario.id}] could not parse task_success judge response")
            return {
                "task_success_score": None,
                "task_success_reason": "unparseable judge response",
            }

    score = parsed.get("task_success_score")
    reason = parsed.get("reason", "")

    logger.info(f"[{scenario.id}] task_success={score} — {reason}")

    return {
        "task_success_score": score,
        "task_success_reason": reason,
    }
