"""
Adapter layer: converts ProbeResult → PipelineResult so all existing
metric functions keep working without any changes.

This is the key architectural trick — metrics expect PipelineResult
with SearchResult sources and GenerationResult. We build those from
the connector's domain-agnostic ProbeResult.
"""

from src.connectors.base import ProbeResult, SourceDocument
from src.database.vector_store import SearchResult
from src.evaluation.runner import Scenario, ScenarioResult
from src.rag.generator import GenerationResult
from src.rag.pipeline import PipelineResult


def _doc_id_to_int(doc_id: str) -> int:
    """
    Metrics like retrieval recall compare source IDs as ints.
    Try to parse the doc_id as int directly; fall back to a
    stable hash so the comparison still works deterministically.
    """
    try:
        return int(doc_id)
    except (ValueError, TypeError):
        return abs(hash(doc_id)) % (10**9)


def _source_doc_to_search_result(doc: SourceDocument) -> SearchResult:
    """Map a domain-agnostic SourceDocument to the SearchResult metrics expect."""
    return SearchResult(
        movie_id=_doc_id_to_int(doc.doc_id),
        title=doc.title,
        chunk_text=doc.content,
        similarity=doc.similarity,
    )


def probe_result_to_pipeline_result(result: ProbeResult) -> PipelineResult:
    """
    Convert a ProbeResult into a PipelineResult that existing metrics
    can score without knowing anything changed.
    """
    sources = [_source_doc_to_search_result(doc) for doc in result.source_documents]

    generation = GenerationResult(
        answer=result.answer,
        model=result.model,
        usage=result.token_usage,
    )

    pipeline_result = PipelineResult(
        question=result.question,
        answer=result.answer,
        sources=sources,
        generation=generation,
        latency_ms=result.latency_ms,
        context_texts=result.context_texts,
    )

    # stash agent trace for tool_selection and task_success metrics
    if result.agent_trace is not None:
        pipeline_result._agent_trace = result.agent_trace  # type: ignore[attr-defined]

    return pipeline_result


def probe_result_to_scenario_result(
    result: ProbeResult,
    scenario: Scenario,
    scores: dict,
) -> ScenarioResult:
    """Build a ScenarioResult from connector output + computed scores."""
    return ScenarioResult(
        scenario=scenario,
        answer=result.answer,
        retrieved_source_ids=[_doc_id_to_int(d.doc_id) for d in result.source_documents],
        retrieved_titles=[d.title for d in result.source_documents],
        similarity_scores=[d.similarity for d in result.source_documents],
        context_texts=result.context_texts,
        latency_ms=result.latency_ms,
        token_usage=result.token_usage,
        model=result.model,
        scores=scores,
    )
