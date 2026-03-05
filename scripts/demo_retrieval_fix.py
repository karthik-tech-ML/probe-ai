"""
Conference demo: diagnosing and fixing a real retrieval failure.

Runs sc_004 ("What is the plot of Avatar?") through both retrieval
strategies side by side:
  - BEFORE: vanilla embedding — Avatar missing from top 5
  - AFTER:  title-boosted embedding — Avatar found

Shows the full eval metrics for both runs to prove measurable improvement.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics.faithfulness import score_faithfulness
from src.evaluation.metrics.grounding import score_grounding
from src.evaluation.metrics.hallucination import score_hallucination
from src.evaluation.metrics.latency import score_latency
from src.evaluation.metrics.retrieval import score_retrieval
from src.evaluation.runner import Scenario, load_scenarios
from src.rag.generator import generate
from src.rag.pipeline import PipelineResult
from src.rag.retriever import retrieve, retrieve_with_title_boost

ALL_METRICS = [
    score_retrieval,
    score_faithfulness,
    score_grounding,
    score_hallucination,
    score_latency,
]

EXPECTED_MOVIE_ID = 19995  # Avatar


def run_pipeline_with_retriever(question: str, retriever_fn, top_k: int = 5) -> PipelineResult:
    """Run the full pipeline but with a swappable retriever function."""
    t0 = time.perf_counter()
    retrieval = retriever_fn(question, top_k=top_k)
    generation = generate(question, retrieval.results)
    latency_ms = (time.perf_counter() - t0) * 1000

    return PipelineResult(
        question=question,
        answer=generation.answer,
        sources=retrieval.results,
        generation=generation,
        latency_ms=round(latency_ms, 1),
    )


def print_results(label: str, scenario: Scenario, result: PipelineResult) -> None:
    """Print retrieval results and eval scores for one run."""
    retrieved_ids = [s.movie_id for s in result.sources]
    avatar_found = EXPECTED_MOVIE_ID in retrieved_ids

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    print(f"\n  Top 5 retrieved:")
    for i, src in enumerate(result.sources, 1):
        marker = " ← EXPECTED" if src.movie_id == EXPECTED_MOVIE_ID else ""
        print(f"    {i}. {src.title} (id={src.movie_id}, sim={src.similarity}){marker}")

    print(f"\n  Avatar in results: {'YES ✓' if avatar_found else 'NO ✗'}")
    print(f"\n  Answer:\n    {result.answer}\n")

    # run all metrics
    scores = {}
    for metric_fn in ALL_METRICS:
        scores.update(metric_fn(scenario, result))

    print(f"  Eval scores:")
    print(f"    Recall@K:      {scores.get('recall_at_k')}")
    print(f"    Faithfulness:  {scores.get('faithfulness_score')}")
    print(f"    Grounding:     {scores.get('grounding_score')}")
    print(f"    Hallucination: {scores.get('hallucination_score')}")
    print(f"    Latency:       {scores.get('latency_ms')}ms "
          f"({'PASS' if scores.get('latency_pass') else 'FAIL'})")


def main() -> None:
    # load sc_004
    scenario = [s for s in load_scenarios() if s.id == "sc_004"][0]

    print(f"\n{'='*60}")
    print(f"  RETRIEVAL FIX DEMO")
    print(f"  Scenario: {scenario.id}")
    print(f"  Question: {scenario.question}")
    print(f"  Expected source: movie_id={EXPECTED_MOVIE_ID} (Avatar)")
    print(f"{'='*60}")

    # BEFORE: vanilla retrieval
    result_before = run_pipeline_with_retriever(scenario.question, retrieve)
    print_results("BEFORE — vanilla embedding", scenario, result_before)

    # AFTER: title-boosted retrieval
    result_after = run_pipeline_with_retriever(scenario.question, retrieve_with_title_boost)
    print_results("AFTER — title-boosted embedding", scenario, result_after)

    # side-by-side comparison
    before_ids = [s.movie_id for s in result_before.sources]
    after_ids = [s.movie_id for s in result_after.sources]

    before_recall = 1.0 if EXPECTED_MOVIE_ID in before_ids else 0.0
    after_recall = 1.0 if EXPECTED_MOVIE_ID in after_ids else 0.0

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"                  BEFORE    AFTER     Δ")
    print(f"  Recall@K:       {before_recall:.1f}       {after_recall:.1f}       {after_recall - before_recall:+.1f}")

    before_top_sim = result_before.sources[0].similarity
    after_top_sim = result_after.sources[0].similarity
    print(f"  Top similarity: {before_top_sim:.4f}    {after_top_sim:.4f}    {after_top_sim - before_top_sim:+.4f}")

    avatar_sim_before = next((s.similarity for s in result_before.sources if s.movie_id == EXPECTED_MOVIE_ID), None)
    avatar_sim_after = next((s.similarity for s in result_after.sources if s.movie_id == EXPECTED_MOVIE_ID), None)
    b_str = f"{avatar_sim_before:.4f}" if avatar_sim_before else "  n/a "
    a_str = f"{avatar_sim_after:.4f}" if avatar_sim_after else "  n/a "
    print(f"  Avatar sim:     {b_str}    {a_str}", end="")
    if avatar_sim_before and avatar_sim_after:
        print(f"    {avatar_sim_after - avatar_sim_before:+.4f}")
    else:
        print()
    print()


if __name__ == "__main__":
    main()
