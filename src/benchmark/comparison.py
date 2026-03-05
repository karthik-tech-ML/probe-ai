"""
Side-by-side comparison of cloud vs local inference results.

Takes BenchmarkResult lists from both backends and produces:
  - Latency stats: average, p50, p95 per backend
  - Cost estimate: Claude API pricing for cloud, $0 for local
  - Quality parity: optional eval metric comparison
  - Printed summary table
"""

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.benchmark.inference import BenchmarkResult


# Claude API pricing (per token, Sonnet 4.5 as of early 2026)
# these are rough — update if the model or pricing changes
_CLOUD_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
_CLOUD_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


@dataclass
class LatencyStats:
    avg_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    count: int


@dataclass
class CostEstimate:
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float


@dataclass
class ComparisonReport:
    """Full comparison between two backends."""

    cloud_latency: LatencyStats | None
    local_latency: LatencyStats | None
    cloud_cost: CostEstimate | None
    local_cost: CostEstimate | None
    # per-scenario answer pairs for quality review
    scenario_pairs: list[dict[str, Any]] = field(default_factory=list)
    # quality scores if eval metrics were run
    cloud_quality: dict[str, float] = field(default_factory=dict)
    local_quality: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.cloud_latency:
            out["cloud_latency"] = {
                "avg_ms": self.cloud_latency.avg_ms,
                "median_ms": self.cloud_latency.median_ms,
                "p95_ms": self.cloud_latency.p95_ms,
                "min_ms": self.cloud_latency.min_ms,
                "max_ms": self.cloud_latency.max_ms,
                "count": self.cloud_latency.count,
            }
        if self.local_latency:
            out["local_latency"] = {
                "avg_ms": self.local_latency.avg_ms,
                "median_ms": self.local_latency.median_ms,
                "p95_ms": self.local_latency.p95_ms,
                "min_ms": self.local_latency.min_ms,
                "max_ms": self.local_latency.max_ms,
                "count": self.local_latency.count,
            }
        if self.cloud_cost:
            out["cloud_cost"] = {
                "input_tokens": self.cloud_cost.total_input_tokens,
                "output_tokens": self.cloud_cost.total_output_tokens,
                "estimated_usd": round(self.cloud_cost.estimated_cost_usd, 6),
            }
        if self.local_cost:
            out["local_cost"] = {
                "input_tokens": self.local_cost.total_input_tokens,
                "output_tokens": self.local_cost.total_output_tokens,
                "estimated_usd": 0.0,
            }
        if self.cloud_quality:
            out["cloud_quality"] = self.cloud_quality
        if self.local_quality:
            out["local_quality"] = self.local_quality
        out["scenario_pairs"] = self.scenario_pairs
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _calc_latency_stats(results: list[BenchmarkResult]) -> LatencyStats | None:
    """Compute latency percentiles from a list of benchmark results."""
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    if not latencies:
        return None
    latencies.sort()
    n = len(latencies)
    # p95 index: grab the value at the 95th percentile position
    p95_idx = min(int(n * 0.95), n - 1)
    return LatencyStats(
        avg_ms=round(statistics.mean(latencies), 1),
        median_ms=round(statistics.median(latencies), 1),
        p95_ms=round(latencies[p95_idx], 1),
        min_ms=round(min(latencies), 1),
        max_ms=round(max(latencies), 1),
        count=n,
    )


def _calc_cost(results: list[BenchmarkResult], is_cloud: bool) -> CostEstimate:
    """Estimate API cost. Local is always $0."""
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    if is_cloud:
        cost = (
            total_in * _CLOUD_INPUT_COST_PER_TOKEN
            + total_out * _CLOUD_OUTPUT_COST_PER_TOKEN
        )
    else:
        cost = 0.0
    return CostEstimate(
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        estimated_cost_usd=round(cost, 6),
    )


def _build_scenario_pairs(
    cloud_results: list[BenchmarkResult],
    local_results: list[BenchmarkResult],
) -> list[dict[str, Any]]:
    """
    Match up scenarios by ID so we can compare answers side by side.
    If only one backend was run, pairs will have None for the missing side.
    """
    cloud_by_id = {r.scenario_id: r for r in cloud_results}
    local_by_id = {r.scenario_id: r for r in local_results}
    all_ids = sorted(set(cloud_by_id.keys()) | set(local_by_id.keys()))

    pairs = []
    for sid in all_ids:
        c = cloud_by_id.get(sid)
        l = local_by_id.get(sid)
        pairs.append({
            "scenario_id": sid,
            "question": (c or l).question,
            "category": (c or l).category,
            "cloud_answer": c.answer if c else None,
            "cloud_latency_ms": c.latency_ms if c else None,
            "cloud_model": c.model if c else None,
            "local_answer": l.answer if l else None,
            "local_latency_ms": l.latency_ms if l else None,
            "local_model": l.model if l else None,
        })
    return pairs


def compare(
    cloud_results: list[BenchmarkResult] | None = None,
    local_results: list[BenchmarkResult] | None = None,
) -> ComparisonReport:
    """
    Build a comparison report from cloud and/or local results.
    Either side can be None if only one backend was run.
    """
    cloud_results = cloud_results or []
    local_results = local_results or []

    report = ComparisonReport(
        cloud_latency=_calc_latency_stats(cloud_results) if cloud_results else None,
        local_latency=_calc_latency_stats(local_results) if local_results else None,
        cloud_cost=_calc_cost(cloud_results, is_cloud=True) if cloud_results else None,
        local_cost=_calc_cost(local_results, is_cloud=False) if local_results else None,
        scenario_pairs=_build_scenario_pairs(cloud_results, local_results),
    )

    return report


def print_comparison(report: ComparisonReport) -> None:
    """Print a human-readable comparison table to stdout."""
    print(f"\n{'='*80}")
    print("  ProbeAI Inference Benchmark — Cloud vs Local")
    print(f"{'='*80}\n")

    # latency comparison
    print("  LATENCY")
    print(f"  {'':20s} {'Cloud':>12s}  {'Local':>12s}")
    print(f"  {'-'*48}")

    def lat(stats, field):
        if stats is None:
            return "       n/a"
        return f"{getattr(stats, field):>8.0f}ms"

    cl = report.cloud_latency
    ll = report.local_latency
    print(f"  {'Average':20s} {lat(cl, 'avg_ms')}  {lat(ll, 'avg_ms')}")
    print(f"  {'Median (p50)':20s} {lat(cl, 'median_ms')}  {lat(ll, 'median_ms')}")
    print(f"  {'p95':20s} {lat(cl, 'p95_ms')}  {lat(ll, 'p95_ms')}")
    print(f"  {'Min':20s} {lat(cl, 'min_ms')}  {lat(ll, 'min_ms')}")
    print(f"  {'Max':20s} {lat(cl, 'max_ms')}  {lat(ll, 'max_ms')}")
    print(f"  {'Scenarios':20s} {(cl.count if cl else 0):>10d}    {(ll.count if ll else 0):>10d}")

    # cost comparison
    print(f"\n  COST")
    cc = report.cloud_cost
    lc = report.local_cost

    def tok(cost, field):
        if cost is None:
            return "       n/a"
        return f"{getattr(cost, field):>10,}"

    print(f"  {'':20s} {'Cloud':>12s}  {'Local':>12s}")
    print(f"  {'-'*48}")
    print(f"  {'Input tokens':20s} {tok(cc, 'total_input_tokens')}  {tok(lc, 'total_input_tokens')}")
    print(f"  {'Output tokens':20s} {tok(cc, 'total_output_tokens')}  {tok(lc, 'total_output_tokens')}")

    cloud_cost_str = f"${cc.estimated_cost_usd:.4f}" if cc else "n/a"
    local_cost_str = "$0.0000" if lc else "n/a"
    print(f"  {'Estimated cost':20s} {cloud_cost_str:>12s}  {local_cost_str:>12s}")

    # quality comparison (if scores were populated)
    if report.cloud_quality or report.local_quality:
        print(f"\n  QUALITY SCORES")
        all_keys = sorted(
            set(report.cloud_quality.keys()) | set(report.local_quality.keys())
        )
        print(f"  {'':20s} {'Cloud':>12s}  {'Local':>12s}")
        print(f"  {'-'*48}")
        for key in all_keys:
            cv = report.cloud_quality.get(key)
            lv = report.local_quality.get(key)
            cs = f"{cv:.2f}" if cv is not None else "n/a"
            ls = f"{lv:.2f}" if lv is not None else "n/a"
            print(f"  {key:20s} {cs:>12s}  {ls:>12s}")

    # per-scenario answers
    print(f"\n  ANSWERS BY SCENARIO")
    print(f"  {'-'*76}")
    for pair in report.scenario_pairs:
        sid = pair["scenario_id"]
        q = pair["question"]
        print(f"\n  [{sid}] {q}")

        if pair.get("cloud_answer"):
            cloud_preview = pair["cloud_answer"][:120].replace("\n", " ")
            cl_ms = pair.get("cloud_latency_ms", 0)
            print(f"    Cloud ({cl_ms:.0f}ms): {cloud_preview}...")
        if pair.get("local_answer"):
            local_preview = pair["local_answer"][:120].replace("\n", " ")
            ll_ms = pair.get("local_latency_ms", 0)
            print(f"    Local ({ll_ms:.0f}ms): {local_preview}...")

    print()
