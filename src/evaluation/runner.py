"""
Eval harness: load scenarios, run each through the RAG pipeline
(or agent or any ProbeConnector), score with metric functions,
output structured JSON results.

Designed to be called from scripts/evaluate.py or used directly
in code. Metric scorers are pluggable — pass any list of functions
that take (scenario, pipeline_result) and return a dict of scores.

Three execution paths:
1. Direct RAG pipeline (default, no connector needed)
2. LangGraph agent (--agent flag)
3. Any ProbeConnector (--connector flag) — the model-agnostic path

All three produce PipelineResult so metrics work unchanged.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from src.rag.pipeline import PipelineResult, ask

# where the scenario library lives by default
DEFAULT_LIBRARY = Path(__file__).resolve().parent.parent.parent / "scenarios" / "library.json"


@dataclass
class Scenario:
    id: str
    question: str
    expected_answer: str
    category: str
    expected_source_ids: list[int]
    difficulty: str
    hallucination_risk: str
    notes: str = ""


@dataclass
class ScenarioResult:
    scenario: Scenario
    answer: str
    retrieved_source_ids: list[int]
    retrieved_titles: list[str]
    similarity_scores: list[float]
    context_texts: list[str]
    latency_ms: float
    token_usage: dict
    model: str
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.id,
            "question": self.scenario.question,
            "category": self.scenario.category,
            "difficulty": self.scenario.difficulty,
            "expected_answer": self.scenario.expected_answer,
            "expected_source_ids": self.scenario.expected_source_ids,
            "answer": self.answer,
            "retrieved_source_ids": self.retrieved_source_ids,
            "retrieved_titles": self.retrieved_titles,
            "similarity_scores": self.similarity_scores,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "model": self.model,
            "scores": self.scores,
        }


# type alias for metric scorer functions
# each takes (scenario, pipeline_result) → dict of metric_name: score
MetricFn = Callable[[Scenario, PipelineResult], dict[str, Any]]


def load_scenarios(
    path: Path = DEFAULT_LIBRARY,
    category: str | None = None,
) -> list[Scenario]:
    """Load scenarios from the JSON library, optionally filtering by category."""
    raw = json.loads(path.read_text())
    scenarios = [Scenario(**s) for s in raw["scenarios"]]

    if category:
        scenarios = [s for s in scenarios if s.category == category]
        logger.info(f"filtered to {len(scenarios)} scenarios in category={category!r}")
    else:
        logger.info(f"loaded {len(scenarios)} scenarios from {path.name}")

    return scenarios


def _run_single(
    scenario: Scenario,
    metrics: list[MetricFn],
    top_k: int,
    model: str,
) -> ScenarioResult:
    """Run one scenario through the pipeline and score it."""
    logger.info(f"[{scenario.id}] {scenario.question}")

    pipeline_result = ask(scenario.question, top_k=top_k, model=model)

    # collect scores from each metric function
    scores: dict[str, Any] = {}
    for metric_fn in metrics:
        try:
            metric_scores = metric_fn(scenario, pipeline_result)
            scores.update(metric_scores)
        except Exception as e:
            logger.error(f"[{scenario.id}] metric {metric_fn.__name__} failed: {e}")
            scores[metric_fn.__name__] = {"error": str(e)}

    return ScenarioResult(
        scenario=scenario,
        answer=pipeline_result.answer,
        retrieved_source_ids=[s.movie_id for s in pipeline_result.sources],
        retrieved_titles=[s.title for s in pipeline_result.sources],
        similarity_scores=[s.similarity for s in pipeline_result.sources],
        context_texts=pipeline_result.context_texts,
        latency_ms=pipeline_result.latency_ms,
        token_usage=pipeline_result.generation.usage,
        model=pipeline_result.generation.model,
        scores=scores,
    )


def _run_single_agent(
    scenario: Scenario,
    metrics: list[MetricFn],
) -> ScenarioResult:
    """
    Run one scenario through the LangGraph agent instead of RAG.

    Wraps the AgentResult into a PipelineResult-shaped object so all
    existing metric functions (faithfulness, grounding, etc.) still
    work unchanged. The agent trace is stashed on the result as
    _agent_trace for tool_selection to pick up.
    """
    from src.agent.graph import run_agent
    from src.database.vector_store import SearchResult
    from src.rag.generator import GenerationResult

    logger.info(f"[{scenario.id}] (agent) {scenario.question}")

    agent_result = run_agent(scenario.question)

    # build a fake PipelineResult so existing metrics still work.
    # the agent doesn't do explicit retrieval — its "context" is the
    # tool outputs that informed the answer.
    context_texts = []
    for tc in agent_result.trace.tool_calls:
        if tc.tool_output:
            ctx = str(tc.tool_output)
            # truncate huge outputs — metrics only need enough to verify claims
            if len(ctx) > 3000:
                ctx = ctx[:3000] + "..."
            context_texts.append(ctx)

    # create a minimal GenerationResult for the interface
    fake_generation = GenerationResult(
        answer=agent_result.answer,
        model="claude-sonnet-4-5-20250929",
        usage={
            "input_tokens": agent_result.trace.total_input_tokens,
            "output_tokens": agent_result.trace.total_output_tokens,
        },
    )

    pipeline_result = PipelineResult(
        question=scenario.question,
        answer=agent_result.answer,
        sources=[],  # agent doesn't do vector retrieval
        generation=fake_generation,
        latency_ms=agent_result.latency_ms,
        context_texts=context_texts,
    )

    # stash the trace so tool_selection and task_success can read it
    pipeline_result._agent_trace = agent_result.trace  # type: ignore[attr-defined]

    # run all metric functions (same interface as RAG path)
    scores: dict[str, Any] = {}
    for metric_fn in metrics:
        try:
            metric_scores = metric_fn(scenario, pipeline_result)
            scores.update(metric_scores)
        except Exception as e:
            logger.error(f"[{scenario.id}] metric {metric_fn.__name__} failed: {e}")
            scores[metric_fn.__name__] = {"error": str(e)}

    return ScenarioResult(
        scenario=scenario,
        answer=agent_result.answer,
        retrieved_source_ids=[],  # no vector retrieval
        retrieved_titles=[],
        similarity_scores=[],
        context_texts=context_texts,
        latency_ms=agent_result.latency_ms,
        token_usage=fake_generation.usage,
        model="claude-sonnet-4-5-20250929 (agent)",
        scores=scores,
    )


def _run_single_connector(
    scenario: Scenario,
    connector: "ProbeConnector",
    metrics: list[MetricFn],
    mode: str = "rag",
) -> ScenarioResult:
    """
    Run one scenario through any ProbeConnector and score it.

    The connector returns a ProbeResult, which the adapter converts
    to PipelineResult so all existing metric functions still work.
    """
    from src.connectors.adapters import (
        probe_result_to_pipeline_result,
        probe_result_to_scenario_result,
    )

    logger.info(f"[{scenario.id}] (connector/{mode}) {scenario.question}")

    probe_result = connector.ask(scenario.question, mode=mode)
    pipeline_result = probe_result_to_pipeline_result(probe_result)

    # run metrics against the adapted PipelineResult — same as always
    scores: dict[str, Any] = {}
    for metric_fn in metrics:
        try:
            metric_scores = metric_fn(scenario, pipeline_result)
            scores.update(metric_scores)
        except Exception as e:
            logger.error(f"[{scenario.id}] metric {metric_fn.__name__} failed: {e}")
            scores[metric_fn.__name__] = {"error": str(e)}

    return probe_result_to_scenario_result(probe_result, scenario, scores)


@dataclass
class EvalRun:
    """Full eval run result — all scenarios + summary stats."""

    results: list[ScenarioResult]
    total_scenarios: int
    total_latency_ms: float
    avg_latency_ms: float
    categories_run: list[str]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_scenarios": self.total_scenarios,
                "total_latency_ms": round(self.total_latency_ms, 1),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "categories_run": self.categories_run,
                "model": self.model,
            },
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def run(
    scenarios: list[Scenario] | None = None,
    category: str | None = None,
    metrics: list[MetricFn] | None = None,
    top_k: int = 5,
    model: str = "claude-sonnet-4-5-20250929",
    use_agent: bool = False,
    connector: "ProbeConnector | None" = None,
) -> EvalRun:
    """
    Run the eval harness across scenarios.

    Args:
        scenarios: pre-loaded list, or None to load from default library
        category: filter to a specific category (ignored if scenarios provided)
        metrics: list of scorer functions. empty list = just collect raw results
        top_k: how many chunks to retrieve per query (RAG mode only)
        model: which Claude model to use for generation (RAG mode only)
        use_agent: if True, route through the LangGraph agent instead of RAG
        connector: if provided, route through this connector instead of
                   the built-in pipeline. overrides use_agent — pass
                   mode via the connector's ask() method instead.
    """
    if scenarios is None:
        scenarios = load_scenarios(category=category)

    if metrics is None:
        metrics = []

    if connector is not None:
        mode_label = "connector"
    elif use_agent:
        mode_label = "agent"
    else:
        mode_label = "RAG"

    logger.info(
        f"starting {mode_label} eval run: {len(scenarios)} scenarios, "
        f"{len(metrics)} metrics"
    )

    # figure out which mode to pass to the connector
    connector_mode = "agent" if use_agent else "rag"

    t0 = time.perf_counter()
    results = []

    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"--- scenario {i}/{len(scenarios)} ---")
        try:
            if connector is not None:
                result = _run_single_connector(
                    scenario, connector, metrics, mode=connector_mode
                )
            elif use_agent:
                result = _run_single_agent(scenario, metrics)
            else:
                result = _run_single(scenario, metrics, top_k, model)
            results.append(result)
        except Exception as e:
            # don't let one crashed scenario kill the whole run —
            # log the error and record a placeholder result
            logger.error(f"[{scenario.id}] scenario failed: {e}")
            results.append(ScenarioResult(
                scenario=scenario,
                answer=f"ERROR: {e}",
                retrieved_source_ids=[],
                retrieved_titles=[],
                similarity_scores=[],
                context_texts=[],
                latency_ms=0,
                token_usage={},
                model=model,
                scores={"error": str(e)},
            ))

    total_ms = (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / len(results) if results else 0
    categories = sorted(set(s.category for s in scenarios))

    eval_run = EvalRun(
        results=results,
        total_scenarios=len(results),
        total_latency_ms=total_ms,
        avg_latency_ms=avg_ms,
        categories_run=categories,
        model=model,
    )

    logger.info(
        f"eval run complete: {len(results)} scenarios in {total_ms:.0f}ms "
        f"(avg {avg_ms:.0f}ms)"
    )

    return eval_run
