"""
Mean Reciprocal Rank (MRR): how early does the first relevant document
appear in the ranked retrieval results?

Formula: MRR = 1 / rank_of_first_relevant_doc

Examples:
  - Relevant doc at position 1: MRR = 1/1 = 1.0
  - Relevant doc at position 3: MRR = 1/3 = 0.33
  - Relevant doc at position 5: MRR = 1/5 = 0.2
  - No relevant doc found:      MRR = 0.0

MRR complements Recall@K: recall tells you *if* the right doc was found,
MRR tells you *where* it landed. A system with perfect recall but MRR=0.2
is still putting the right document at position 5 — the generator has to
wade through noise before it gets to the good stuff.

Purely mechanical — no LLM calls. Uses the same expected_source_ids field
as recall and precision.
"""

from typing import Any

from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult


def score_mrr(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    Mean Reciprocal Rank for the ranked retrieval list.

    Walks retrieved docs from position 1 to K and returns 1/position
    at the first relevant hit. Returns 0.0 if no relevant doc was
    retrieved at all.

    Skips scoring (returns None) when expected_source_ids is empty —
    no_answer scenarios have no "right" document to find, so MRR=0.0
    would look like a failure when the system is doing the right thing.

    Returns:
      mrr: 0.0-1.0, or None if not applicable
      mrr_first_hit_position: rank of the first relevant doc (1-indexed), or None
    """
    expected = set(scenario.expected_source_ids)
    retrieved = [s.movie_id for s in result.sources]

    # no ground truth IDs — skip rather than report misleading zeros
    if not expected:
        return {"mrr": None, "mrr_first_hit_position": None}

    # agent path (no vector retrieval) or empty result
    if not retrieved:
        return {"mrr": 0.0, "mrr_first_hit_position": None}

    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected:
            return {
                "mrr": round(1.0 / position, 4),
                "mrr_first_hit_position": position,
            }

    # relevant doc wasn't retrieved at all
    return {"mrr": 0.0, "mrr_first_hit_position": None}
