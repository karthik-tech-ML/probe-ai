"""
Retrieval quality metrics: did we pull back the right documents?

These score how well the vector search found the expected source
movies, without caring about the generated answer quality.
"""

from typing import Any

from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult


def score_retrieval(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    Compute retrieval recall and precision against expected source IDs.

    Recall@K: what fraction of expected sources showed up in top-K?
      → 1.0 means we found every expected source
      → 0.0 means we missed all of them

    Precision@K: what fraction of retrieved results are expected sources?
      → high precision = not much noise in the results

    Skips scoring when expected_source_ids is empty (e.g. no_answer
    scenarios where there's no "right" document to retrieve).
    """
    expected = set(scenario.expected_source_ids)
    retrieved = [s.movie_id for s in result.sources]
    retrieved_set = set(retrieved)

    # nothing to measure against — skip rather than return misleading zeros
    if not expected:
        return {
            "recall_at_k": None,
            "precision_at_k": None,
            "expected_found": [],
            "expected_missing": [],
        }

    found = expected & retrieved_set
    missing = expected - retrieved_set

    recall = len(found) / len(expected)
    precision = len(found) / len(retrieved) if retrieved else 0.0

    # where did the expected docs land in the ranking?
    # useful for debugging — maybe the doc was retrieved but at position 5
    hit_positions = {
        mid: retrieved.index(mid) + 1
        for mid in found
    }

    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "expected_found": sorted(found),
        "expected_missing": sorted(missing),
        "hit_positions": hit_positions,
    }
