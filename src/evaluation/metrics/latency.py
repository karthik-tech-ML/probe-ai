"""
Latency pass/fail check against the configured threshold.

No LLM call needed — just compares pipeline latency_ms against
LATENCY_THRESHOLD_MS from settings.
"""

from typing import Any

from src.config import settings
from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult


def score_latency(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    Check if the pipeline response time is under the threshold.

    Returns:
      latency_pass: bool
      latency_ms: actual latency
      latency_threshold_ms: what we measured against
    """
    threshold = settings.latency_threshold_ms
    actual = result.latency_ms
    passed = actual <= threshold

    return {
        "latency_pass": passed,
        "latency_ms": round(actual, 1),
        "latency_threshold_ms": threshold,
    }
