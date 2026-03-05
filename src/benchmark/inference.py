"""
Benchmark runner: push scenarios through a specified inference backend
and collect latency, token counts, and answer text.

Supports two backends:
  - "cloud": Claude API via the existing RAG generator
  - "local": Ollama via local_backend.py

Both use the same retrieval step (pgvector) so the only variable is
the generation model. This lets us do an honest comparison of answer
quality and speed.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from loguru import logger

from src.benchmark.local_backend import (
    DEFAULT_MODEL as DEFAULT_LOCAL_MODEL,
    generate as local_generate,
)
from src.evaluation.runner import Scenario
from src.rag.retriever import retrieve


@dataclass
class BenchmarkResult:
    """One scenario run through one backend."""

    scenario_id: str
    question: str
    category: str
    backend: str  # "cloud" or "local"
    model: str
    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    # context that was fed to the model (for eval scoring later)
    context_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "question": self.question,
            "category": self.category,
            "backend": self.backend,
            "model": self.model,
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def _run_cloud(
    question: str,
    context_texts: list[str],
    model: str = "claude-sonnet-4-5-20250929",
) -> BenchmarkResult:
    """Generate an answer via Claude API."""
    from src.database.vector_store import SearchResult
    from src.rag.generator import generate

    # generator expects SearchResult objects, but we only need chunk_text.
    # build minimal shims so we don't duplicate the prompt-building logic.
    fake_chunks = [
        SearchResult(movie_id=0, title="", chunk_text=text, similarity=0.0)
        for text in context_texts
    ]

    t0 = time.perf_counter()
    gen_result = generate(question, fake_chunks, model=model)
    latency_ms = (time.perf_counter() - t0) * 1000

    return BenchmarkResult(
        scenario_id="",  # filled in by caller
        question=question,
        category="",
        backend="cloud",
        model=gen_result.model,
        answer=gen_result.answer,
        latency_ms=round(latency_ms, 1),
        input_tokens=gen_result.usage.get("input_tokens", 0),
        output_tokens=gen_result.usage.get("output_tokens", 0),
        context_texts=context_texts,
    )


def _run_local(
    question: str,
    context_texts: list[str],
    model: str = DEFAULT_LOCAL_MODEL,
) -> BenchmarkResult:
    """Generate an answer via Ollama."""
    result = local_generate(question, context_texts, model=model)

    return BenchmarkResult(
        scenario_id="",
        question=question,
        category="",
        backend="local",
        model=result.model,
        answer=result.answer,
        latency_ms=result.latency_ms,
        input_tokens=result.prompt_eval_count,
        output_tokens=result.eval_count,
        context_texts=context_texts,
    )


def run_benchmark(
    scenarios: list[Scenario],
    backend: Literal["cloud", "local"],
    top_k: int = 5,
    model: str | None = None,
) -> list[BenchmarkResult]:
    """
    Run a list of scenarios through a single backend.

    Retrieval always goes through pgvector — we're only swapping out
    the generation step. This isolates the variable we actually care
    about: how does the LLM perform given the same retrieved context?
    """
    if model is None:
        model = (
            "claude-sonnet-4-5-20250929" if backend == "cloud"
            else DEFAULT_LOCAL_MODEL
        )

    runner = _run_cloud if backend == "cloud" else _run_local
    results: list[BenchmarkResult] = []

    logger.info(
        f"starting benchmark: {len(scenarios)} scenarios, "
        f"backend={backend}, model={model}"
    )

    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"[{i}/{len(scenarios)}] {scenario.id}: {scenario.question}")

        try:
            # shared retrieval — both backends get the same chunks
            retrieval = retrieve(scenario.question, top_k=top_k)
            context_texts = [r.chunk_text for r in retrieval.results]

            result = runner(scenario.question, context_texts, model=model)
            result.scenario_id = scenario.id
            result.category = scenario.category
            result.context_texts = context_texts

            results.append(result)
            logger.info(
                f"  → {result.latency_ms:.0f}ms, "
                f"{result.input_tokens}in/{result.output_tokens}out tokens"
            )

        except Exception as e:
            logger.error(f"[{scenario.id}] benchmark failed: {e}")
            results.append(BenchmarkResult(
                scenario_id=scenario.id,
                question=scenario.question,
                category=scenario.category,
                backend=backend,
                model=model,
                answer=f"ERROR: {e}",
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
            ))

    return results
