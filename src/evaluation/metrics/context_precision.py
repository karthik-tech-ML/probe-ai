"""
Context Precision (Precision@K): of the K retrieved documents,
how many were actually relevant?

This is the complement to Recall@K in retrieval.py:
  - Recall@K asks: did we retrieve all the documents we should have?
  - Context Precision asks: did we only retrieve documents we should have?

High recall + low precision = the right docs showed up but so did a lot of noise.
High precision + low recall = everything retrieved was relevant, but we missed some.
Both matter. This file exists so Context Precision shows up as its own named
metric in results — aligned with RAGAS vocabulary — rather than being buried
inside the retrieval scorer's output.

Mechanical scoring — no LLM needed. Uses expected_source_ids from the scenario,
same as score_retrieval. Returns None when expected_source_ids is empty (e.g.
no_answer scenarios where there's no "right" document to retrieve), or when no
documents were retrieved (agent path).
"""

from typing import Any

from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult


def score_context_precision(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    Precision@K: what fraction of the K retrieved documents were relevant?

    If K=5 and 3 of the 5 match expected_source_ids, precision = 0.6.

    Returns None rather than zero when there are no expected_source_ids —
    returning 0.0 for no_answer scenarios would look like failures when
    they're actually working correctly.

    Returns:
      context_precision: 0.0-1.0, or None if not applicable
    """
    expected = set(scenario.expected_source_ids)
    retrieved = [s.movie_id for s in result.sources]

    # no ground truth IDs to measure against — skip rather than mislead
    if not expected:
        return {"context_precision": None}

    # agent path doesn't do vector retrieval — not applicable
    if not retrieved:
        return {"context_precision": None}

    relevant_count = sum(1 for mid in retrieved if mid in expected)
    precision = round(relevant_count / len(retrieved), 4)

    return {"context_precision": precision}
