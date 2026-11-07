"""
Normalized Discounted Cumulative Gain (nDCG): are the most relevant
documents ranked near the top of the retrieval results?

nDCG goes further than MRR and Precision@K:
  - Unlike MRR: cares about ALL relevant docs, not just the first one
  - Unlike Precision@K: gives diminishing credit to lower-ranked positions
  - Normalized to 0–1 so scores are comparable across queries

Formula:
  DCG  = Σ relevance_i / log2(position_i + 1)    for i in 1..K
  IDCG = DCG for the perfect ranking (all relevant docs ranked first)
  nDCG = DCG / IDCG

Relevance is binary in our case: 1 if doc ID is in expected_source_ids,
0 otherwise. This matches how most IR benchmarks use nDCG when you don't
have graded relevance labels.

Example with K=5, 2 relevant docs expected:
  Retrieved:  [rel, irrel, rel, irrel, irrel]
  DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0 + 0
       = 1.0 + 0.5 = 1.5
  IDCG = 1/log2(2) + 1/log2(3)        (ideal: both relevant docs first)
       = 1.0 + 0.631 = 1.631
  nDCG = 1.5 / 1.631 = 0.92

Purely mechanical — no LLM calls. Uses expected_source_ids same as
recall, precision, and MRR.
"""

import math
from typing import Any

from src.evaluation.runner import Scenario
from src.rag.pipeline import PipelineResult


def _dcg(relevances: list[int]) -> float:
    """
    Discounted Cumulative Gain for a ranked relevance list.

    Each position contributes rel / log2(position + 1), so position 1
    gets full credit, position 2 gets 63%, position 4 gets 50%, etc.
    """
    total = 0.0
    for position, rel in enumerate(relevances, start=1):
        if rel > 0:
            total += rel / math.log2(position + 1)
    return total


def score_ndcg(scenario: Scenario, result: PipelineResult) -> dict[str, Any]:
    """
    nDCG for the ranked retrieval list.

    Assigns binary relevance (1/0) to each retrieved position, computes
    DCG, then normalizes against IDCG — the DCG you'd get if all relevant
    docs were ranked first.

    Skips scoring (returns None) when expected_source_ids is empty —
    no_answer scenarios have no "right" doc to rank, so nDCG=0.0 would
    be misleading.

    Returns:
      ndcg: 0.0-1.0, or None if not applicable
    """
    expected = set(scenario.expected_source_ids)
    retrieved = [s.movie_id for s in result.sources]

    # no ground truth IDs — skip rather than report misleading zeros
    if not expected:
        return {"ndcg": None}

    # agent path (no vector retrieval) or empty result
    if not retrieved:
        return {"ndcg": 0.0}

    # binary relevance at each retrieved position
    relevances = [1 if doc_id in expected else 0 for doc_id in retrieved]

    dcg = _dcg(relevances)

    # IDCG: ideal ranking where all relevant docs come first.
    # Bounded by min(num_expected, K) — we can't put more relevant docs
    # at the top than either exist in the corpus or were retrieved.
    num_relevant = len(expected)
    ideal_hits = min(num_relevant, len(retrieved))
    ideal_relevances = [1] * ideal_hits + [0] * (len(retrieved) - ideal_hits)
    idcg = _dcg(ideal_relevances)

    # idcg can only be 0 if ideal_hits is 0, which can't happen here
    # since we already checked expected is non-empty and retrieved is non-empty.
    # guard anyway for safety.
    if idcg == 0.0:
        return {"ndcg": 0.0}

    return {"ndcg": round(dcg / idcg, 4)}
