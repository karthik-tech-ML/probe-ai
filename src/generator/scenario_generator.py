"""
Auto-generate evaluation scenarios from any connected system.

Takes a ProbeConnector, pulls its schema and sample data, then
uses Claude to generate category-specific test scenarios. The
output matches the scenarios/library.json format so it can be
fed directly into the eval runner.
"""

import json
from typing import Any

import anthropic
from loguru import logger

from src.config import settings
from src.connectors.base import ProbeConnector
from src.generator.prompts import ALL_CATEGORIES, CATEGORY_PROMPTS


def _call_claude(prompt: str, model: str) -> str:
    """Send a prompt to Claude and return the raw text response."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _parse_scenarios(raw_text: str) -> list[dict[str, Any]]:
    """
    Parse Claude's JSON response into scenario dicts.

    Handles the common case where Claude wraps JSON in markdown
    code fences even though we asked it not to.
    """
    text = raw_text.strip()

    # strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # drop first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    scenarios = json.loads(text)

    if not isinstance(scenarios, list):
        raise ValueError(f"Expected a JSON array, got {type(scenarios).__name__}")

    return scenarios


def _validate_scenario(scenario: dict, category: str) -> dict | None:
    """
    Check that a generated scenario has all required fields.
    Returns the cleaned scenario or None if it's malformed.
    """
    required = {"id", "question", "expected_answer", "category",
                "expected_source_ids", "difficulty", "hallucination_risk"}

    missing = required - set(scenario.keys())
    if missing:
        logger.warning(f"skipping scenario {scenario.get('id', '?')}: missing {missing}")
        return None

    # enforce the category we asked for
    scenario["category"] = category

    # make sure expected_source_ids is a list
    if not isinstance(scenario["expected_source_ids"], list):
        scenario["expected_source_ids"] = []

    # add notes if missing
    if "notes" not in scenario:
        scenario["notes"] = ""

    return scenario


def generate_scenarios(
    connector: ProbeConnector,
    categories: list[str] | None = None,
    count_per_category: int = 5,
    model: str = "claude-sonnet-4-5-20250929",
) -> list[dict[str, Any]]:
    """
    Generate test scenarios for any connected system.

    Pulls the schema and sample data from the connector, then
    asks Claude to create scenarios for each requested category.

    Args:
        connector: the system to generate scenarios for
        categories: which categories to generate (default: all)
        count_per_category: how many scenarios per category
        model: which Claude model to use for generation

    Returns:
        List of scenario dicts matching the library.json schema.
    """
    if categories is None:
        categories = ALL_CATEGORIES

    # validate categories
    invalid = [c for c in categories if c not in CATEGORY_PROMPTS]
    if invalid:
        available = ", ".join(ALL_CATEGORIES)
        raise ValueError(f"Unknown categories: {invalid}. Available: {available}")

    logger.info(f"generating scenarios for {len(categories)} categories, {count_per_category} each")

    # get data from the connector
    schema = connector.get_schema()
    samples = connector.get_corpus_sample(n=10)

    logger.info(f"corpus: {schema.domain_name}, {schema.total_documents} documents")

    all_scenarios: list[dict[str, Any]] = []
    scenario_counter = 1

    for category in categories:
        logger.info(f"generating {count_per_category} '{category}' scenarios...")

        prompt_fn = CATEGORY_PROMPTS[category]
        prompt = prompt_fn(schema, samples, count_per_category)

        try:
            raw = _call_claude(prompt, model)
            parsed = _parse_scenarios(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"failed to parse {category} scenarios: {e}")
            continue

        for scenario in parsed:
            validated = _validate_scenario(scenario, category)
            if validated is None:
                continue

            # re-number IDs sequentially across all categories
            validated["id"] = f"gen_{scenario_counter:03d}"
            scenario_counter += 1
            all_scenarios.append(validated)

        logger.info(f"  got {len(parsed)} scenarios for '{category}'")

    logger.info(f"total generated: {len(all_scenarios)} scenarios")
    return all_scenarios


def scenarios_to_library(
    scenarios: list[dict[str, Any]],
    domain_name: str = "",
) -> dict[str, Any]:
    """
    Wrap scenario list in the library.json format with version
    and description metadata.
    """
    return {
        "version": "1.0",
        "description": f"ProbeAI Generated Scenarios — {domain_name}" if domain_name
                       else "ProbeAI Generated Scenarios",
        "scenarios": scenarios,
    }
