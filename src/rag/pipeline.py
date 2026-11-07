"""
End-to-end RAG pipeline: question → retrieve → generate → answer.

This is the main entry point for asking questions. It wires together
the retriever and generator, tracks latency, and returns everything
the eval harness needs to score the response.
"""

import time
from dataclasses import dataclass, field

from loguru import logger

from src.config import settings
from src.database.vector_store import SearchResult
from src.rag.generator import GenerationResult, generate
from src.rag.retriever import retrieve, retrieve_with_title_boost


@dataclass
class PipelineResult:
    question: str
    answer: str
    sources: list[SearchResult]
    generation: GenerationResult
    latency_ms: float
    # for eval: easy access to just the context texts
    context_texts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.context_texts:
            self.context_texts = [s.chunk_text for s in self.sources]


def ask(
    question: str,
    top_k: int = 5,
    model: str = settings.llm_model,
    title_boost: bool = True,
) -> PipelineResult:
    """
    Full RAG pipeline: embed the question, retrieve relevant chunks,
    generate an answer grounded in those chunks.

    title_boost: when True (default), uses retrieve_with_title_boost()
      which mirrors the chunk template format so exact-title queries
      land near their document instead of near thematically similar ones.
      Example: "What is the plot of Avatar?" without boost retrieves
      Princess Mononoke (sim=0.45); with boost retrieves Avatar (sim=0.72).

    Returns everything needed for evaluation: answer, sources with
    similarity scores, token usage, and total latency.
    """
    t0 = time.perf_counter()

    # 1. retrieve — use title-boost by default to fix exact-title drift
    if title_boost:
        retrieval = retrieve_with_title_boost(question, top_k=top_k)
    else:
        retrieval = retrieve(question, top_k=top_k)

    # 2. generate
    generation = generate(question, retrieval.results, model=model)

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(f"pipeline done in {latency_ms:.0f}ms")

    return PipelineResult(
        question=question,
        answer=generation.answer,
        sources=retrieval.results,
        generation=generation,
        latency_ms=round(latency_ms, 1),
    )
