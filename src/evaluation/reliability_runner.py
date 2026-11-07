"""
Pass@K Reliability Runner.

Measures whether the pipeline produces consistently correct answers
across K independent runs of the same question. This surfaces two
classes of failure that a single-shot eval misses:

1. Flaky passes  — the answer is correct 1 out of K times by chance.
2. Flaky fails   — the answer is correct K-1 out of K times; the system
                   is almost reliable but not trustworthy in production.

Key metrics per scenario:
  pass_rate        — fraction of K runs that passed (0.0 – 1.0)
  strict_pass      — True if every single run passed (pass_rate == 1.0)
  consistency_score — 1 – stdev(composite_scores); measures how stable
                      the quality signal is across runs (0.0 – 1.0)
  score_variance   — variance of per-run composite quality scores

A run "passes" when:
  faithfulness_score >= pass_threshold  AND  answer_relevancy >= pass_threshold

For no_answer scenarios where faithfulness may be None, only
answer_relevancy is required.

Only lightweight metrics are run per attempt (faithfulness, answer_relevancy,
latency, cost). Heavier metrics (grounding, hallucination, retrieval) are
omitted to keep K×N API calls manageable.
"""

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from src.connectors.adapters import probe_result_to_pipeline_result
from src.evaluation.metrics.answer_relevancy import score_answer_relevancy
from src.evaluation.metrics.cost_per_query import score_cost_per_query
from src.evaluation.metrics.faithfulness import score_faithfulness
from src.evaluation.metrics.latency import score_latency
from src.evaluation.runner import Scenario, load_scenarios

# default pass threshold — a run is "correct" if both LLM judge scores >= this
DEFAULT_PASS_THRESHOLD = 0.7


@dataclass
class RunAttempt:
    """Single execution of one scenario."""
    run_index: int                  # 1-based
    answer: str
    faithfulness: float | None      # None if judge errored or no_answer
    answer_relevancy: float | None  # None if judge errored
    latency_ms: float
    total_tokens: int
    cost_usd: float
    passed: bool                    # True if both scores >= pass_threshold


@dataclass
class ReliabilityResult:
    """Aggregated reliability metrics across K runs of one scenario."""
    scenario_id: str
    question: str
    category: str
    difficulty: str
    k: int                          # number of runs attempted
    pass_threshold: float

    # per-attempt data
    attempts: list[RunAttempt] = field(default_factory=list)

    # aggregate metrics — populated by _compute_aggregates()
    passes: int = 0
    pass_rate: float = 0.0
    strict_pass: bool = False
    consistency_score: float | None = None   # None if k < 2 or all scores None
    score_variance: float | None = None
    avg_faithfulness: float | None = None
    avg_answer_relevancy: float | None = None
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    avg_cost_usd: float = 0.0

    def compute_aggregates(self) -> None:
        """Fill aggregate fields from the attempts list. Call after all runs."""
        if not self.attempts:
            return

        self.passes = sum(1 for a in self.attempts if a.passed)
        self.pass_rate = round(self.passes / len(self.attempts), 4)
        self.strict_pass = self.passes == len(self.attempts)

        # composite score per run = mean of non-None judge scores
        composites: list[float] = []
        for a in self.attempts:
            vals = [v for v in [a.faithfulness, a.answer_relevancy] if v is not None]
            if vals:
                composites.append(sum(vals) / len(vals))

        if len(composites) >= 2:
            stdev = statistics.stdev(composites)
            self.consistency_score = round(max(0.0, 1.0 - stdev), 4)
            self.score_variance = round(statistics.variance(composites), 6)
        elif len(composites) == 1:
            self.consistency_score = 1.0   # can't measure with a single data point
            self.score_variance = 0.0

        faith_vals = [a.faithfulness for a in self.attempts if a.faithfulness is not None]
        rel_vals   = [a.answer_relevancy for a in self.attempts if a.answer_relevancy is not None]
        if faith_vals:
            self.avg_faithfulness = round(sum(faith_vals) / len(faith_vals), 4)
        if rel_vals:
            self.avg_answer_relevancy = round(sum(rel_vals) / len(rel_vals), 4)

        self.avg_latency_ms = round(
            sum(a.latency_ms for a in self.attempts) / len(self.attempts), 1
        )
        self.total_cost_usd = round(sum(a.cost_usd for a in self.attempts), 6)
        self.avg_cost_usd   = round(self.total_cost_usd / len(self.attempts), 6)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ReliabilityReport:
    """Full reliability run across all sampled scenarios."""
    scenarios_tested: int
    k: int
    pass_threshold: float
    total_runs: int
    total_cost_usd: float
    total_time_ms: float
    results: list[ReliabilityResult] = field(default_factory=list)

    # aggregate summary
    avg_pass_rate: float = 0.0
    pct_strict_pass: float = 0.0      # % of scenarios with strict_pass=True
    avg_consistency: float | None = None

    def compute_summary(self) -> None:
        if not self.results:
            return
        self.avg_pass_rate = round(
            sum(r.pass_rate for r in self.results) / len(self.results), 4
        )
        self.pct_strict_pass = round(
            sum(1 for r in self.results if r.strict_pass) / len(self.results), 4
        )
        consistencies = [r.consistency_score for r in self.results
                         if r.consistency_score is not None]
        if consistencies:
            self.avg_consistency = round(sum(consistencies) / len(consistencies), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "scenarios_tested": self.scenarios_tested,
                "k": self.k,
                "pass_threshold": self.pass_threshold,
                "total_runs": self.total_runs,
                "total_cost_usd": round(self.total_cost_usd, 4),
                "total_time_ms": round(self.total_time_ms, 1),
                "avg_pass_rate": self.avg_pass_rate,
                "pct_strict_pass": self.pct_strict_pass,
                "avg_consistency": self.avg_consistency,
            },
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── core runner ───────────────────────────────────────────────────────────────

def _run_one_attempt(
    scenario: Scenario,
    connector: Any,
    pass_threshold: float,
    run_index: int,
) -> RunAttempt:
    """Execute one attempt of a scenario and return scored result."""
    probe_result = connector.ask(scenario.question, mode="rag")
    pipeline_result = probe_result_to_pipeline_result(probe_result)

    # lightweight metric set — just what we need for pass/fail + cost
    faith_scores = {}
    rel_scores   = {}
    lat_scores   = {}
    cost_scores  = {}

    try:
        faith_scores = score_faithfulness(scenario, pipeline_result)
    except Exception as e:
        logger.warning(f"[{scenario.id}] run {run_index}: faithfulness scorer failed: {e}")

    try:
        rel_scores = score_answer_relevancy(scenario, pipeline_result)
    except Exception as e:
        logger.warning(f"[{scenario.id}] run {run_index}: answer_relevancy scorer failed: {e}")

    try:
        lat_scores = score_latency(scenario, pipeline_result)
    except Exception as e:
        logger.warning(f"[{scenario.id}] run {run_index}: latency scorer failed: {e}")

    try:
        cost_scores = score_cost_per_query(scenario, pipeline_result)
    except Exception as e:
        logger.warning(f"[{scenario.id}] run {run_index}: cost scorer failed: {e}")

    faithfulness      = faith_scores.get("faithfulness_score")
    answer_relevancy  = rel_scores.get("answer_relevancy")
    cost_usd          = cost_scores.get("cost_usd", 0.0)
    latency_ms        = pipeline_result.latency_ms

    # token count from pipeline result
    usage = pipeline_result.generation.usage or {}
    total_tokens = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
    if total_tokens == 0:
        # fallback estimate
        total_tokens = max(1, (len(pipeline_result.answer) + sum(
            len(t) for t in pipeline_result.context_texts
        )) // 4)

    # pass logic: need both judge scores >= threshold
    # for no_answer scenarios faithfulness may be None — use relevancy alone
    if faithfulness is not None and answer_relevancy is not None:
        passed = faithfulness >= pass_threshold and answer_relevancy >= pass_threshold
    elif answer_relevancy is not None:
        passed = answer_relevancy >= pass_threshold
    else:
        passed = False   # can't judge — conservative: mark as fail

    logger.info(
        f"[{scenario.id}] run {run_index}: "
        f"faith={faithfulness!r} rel={answer_relevancy!r} "
        f"{'PASS' if passed else 'FAIL'} "
        f"({latency_ms:.0f}ms, ${cost_usd:.4f})"
    )

    return RunAttempt(
        run_index=run_index,
        answer=pipeline_result.answer,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        latency_ms=latency_ms,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        passed=passed,
    )


def run_reliability(
    scenarios: list[Scenario],
    connector: Any,
    k: int = 5,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    sample: int | None = None,
    seed: int = 42,
) -> ReliabilityReport:
    """
    Run each scenario K times and compute pass@k reliability metrics.

    Args:
        scenarios: pre-loaded scenario list
        connector:  any ProbeConnector instance (tmdb, etc.)
        k:          number of independent runs per scenario
        pass_threshold: minimum score for faithfulness + answer_relevancy
                        to count as a passing run (default 0.7)
        sample:     if set, randomly select this many scenarios instead of all
        seed:       RNG seed for reproducible sampling

    Returns:
        ReliabilityReport with per-scenario results and aggregate summary
    """
    if sample is not None and sample < len(scenarios):
        rng = random.Random(seed)
        scenarios = rng.sample(scenarios, sample)
        logger.info(f"sampled {sample} scenarios (seed={seed})")

    logger.info(
        f"reliability run: {len(scenarios)} scenarios × {k} runs = "
        f"{len(scenarios) * k} total API calls"
    )

    t0 = time.perf_counter()
    results: list[ReliabilityResult] = []

    for scenario_num, scenario in enumerate(scenarios, 1):
        logger.info(
            f"=== scenario {scenario_num}/{len(scenarios)}: "
            f"[{scenario.id}] {scenario.question!r} ==="
        )

        rel_result = ReliabilityResult(
            scenario_id=scenario.id,
            question=scenario.question,
            category=scenario.category,
            difficulty=scenario.difficulty,
            k=k,
            pass_threshold=pass_threshold,
        )

        for run_index in range(1, k + 1):
            logger.debug(f"  run {run_index}/{k}")
            try:
                attempt = _run_one_attempt(scenario, connector, pass_threshold, run_index)
                rel_result.attempts.append(attempt)
            except Exception as e:
                logger.error(f"[{scenario.id}] run {run_index} crashed: {e}")
                # record a failed attempt so we don't silently drop runs
                rel_result.attempts.append(RunAttempt(
                    run_index=run_index,
                    answer=f"ERROR: {e}",
                    faithfulness=None,
                    answer_relevancy=None,
                    latency_ms=0.0,
                    total_tokens=0,
                    cost_usd=0.0,
                    passed=False,
                ))

        rel_result.compute_aggregates()
        results.append(rel_result)

        logger.info(
            f"  → pass_rate={rel_result.pass_rate:.2f} "
            f"({rel_result.passes}/{k}) "
            f"strict={rel_result.strict_pass} "
            f"consistency={rel_result.consistency_score}"
        )

    total_ms = (time.perf_counter() - t0) * 1000
    total_cost = sum(r.total_cost_usd for r in results)

    report = ReliabilityReport(
        scenarios_tested=len(results),
        k=k,
        pass_threshold=pass_threshold,
        total_runs=sum(len(r.attempts) for r in results),
        total_cost_usd=total_cost,
        total_time_ms=total_ms,
        results=results,
    )
    report.compute_summary()

    logger.info(
        f"reliability run complete: "
        f"{report.total_runs} runs in {total_ms / 1000:.1f}s, "
        f"avg_pass_rate={report.avg_pass_rate:.2f}, "
        f"strict_pass={report.pct_strict_pass:.0%}, "
        f"total_cost=${total_cost:.4f}"
    )

    return report
