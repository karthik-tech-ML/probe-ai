"""
Pass@K Reliability Runner — CLI.

Runs each scenario K times to measure how consistently the pipeline
produces correct answers. A single-shot eval can't tell you whether a
score of 0.8 means "reliably good" or "50% chance of getting lucky."
This runner answers that question.

Usage:
    # small smoke test — 5 scenarios × 3 runs = 15 API calls (~$0.07)
    python scripts/reliability.py --connector tmdb --k 3 --sample 5

    # full test — 10 scenarios × 8 runs = 80 API calls (~$0.40)
    python scripts/reliability.py --connector tmdb --k 8 --sample 10

    # specific category only
    python scripts/reliability.py --connector tmdb --k 5 --category simple_factual

    # lower pass threshold (default 0.7)
    python scripts/reliability.py --connector tmdb --k 5 --pass-threshold 0.8

    # save JSON results
    python scripts/reliability.py --connector tmdb --k 5 --sample 10 --out results/reliability_k5.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.connectors import get_connector
from src.evaluation.reliability_runner import (
    DEFAULT_PASS_THRESHOLD,
    ReliabilityReport,
    ReliabilityResult,
    run_reliability,
)
from src.evaluation.runner import load_scenarios


# ── formatting helpers ────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    if v is None:
        return "  n/a"
    return f"{v * 100:5.1f}%"


def _score(v: float | None) -> str:
    if v is None:
        return "   n/a"
    return f"  {v:.3f}"


def _ms(v: float) -> str:
    return f"{v:>6.0f}ms"


def _cost(v: float) -> str:
    return f"${v:.4f}"


def _passes(passes: int, k: int) -> str:
    return f"{passes}/{k}"


# ── report printing ───────────────────────────────────────────────────────────

def print_report(report: ReliabilityReport) -> None:
    k = report.k
    threshold = report.pass_threshold

    print(f"\n{'=' * 140}")
    print(
        f"  ProbeAI Reliability Results — "
        f"{report.scenarios_tested} scenarios × {k} runs = {report.total_runs} total | "
        f"pass_threshold={threshold:.1f} | "
        f"total_cost=${report.total_cost_usd:.4f} | "
        f"time={report.total_time_ms / 1000:.1f}s"
    )
    print(f"{'=' * 140}\n")

    # header
    hdr = (
        f"  {'ID':<8} {'Category':<17} {'Question':<40} "
        f"{'Pass/K':>7} {'Rate':>7} {'Strict':>7} "
        f"{'Consist':>8} {'Variance':>9} "
        f"{'Avg Faith':>10} {'Avg Rel':>8} "
        f"{'AvgLat':>10} {'AvgCost':>9}"
    )
    print(hdr)
    print(f"  {'-' * (len(hdr.strip()) + 2)}")

    for r in report.results:
        q_short = (r.question[:37] + "...") if len(r.question) > 40 else r.question
        strict_sym = " ✓" if r.strict_pass else " ✗"
        print(
            f"  {r.scenario_id:<8} {r.category:<17} {q_short:<40} "
            f"  {_passes(r.passes, r.k):>5} "
            f"{_pct(r.pass_rate):>7} "
            f"{strict_sym:>7} "
            f"{_score(r.consistency_score):>8} "
            f"{_score(r.score_variance):>9} "
            f"{_score(r.avg_faithfulness):>10} "
            f"{_score(r.avg_answer_relevancy):>8} "
            f"  {_ms(r.avg_latency_ms):>10} "
            f"{_cost(r.avg_cost_usd):>9}"
        )

    # per-run breakdown section
    print(f"\n  {'— Per-Run Detail —':^130}\n")
    for r in report.results:
        print(f"  {r.scenario_id} [{r.category}] {r.question[:70]}")
        for a in r.attempts:
            faith_str = f"{a.faithfulness:.2f}" if a.faithfulness is not None else " n/a"
            rel_str   = f"{a.answer_relevancy:.2f}" if a.answer_relevancy is not None else " n/a"
            sym = "✓" if a.passed else "✗"
            print(
                f"    run {a.run_index:>2}: {sym}  "
                f"faith={faith_str}  rel={rel_str}  "
                f"{a.latency_ms:>6.0f}ms  ${a.cost_usd:.4f}  "
                f"{a.total_tokens:>5} tok"
            )
        print()

    # summary
    print(f"  {'— Summary —':^130}")
    print(f"  Avg Pass Rate:       {report.avg_pass_rate * 100:.1f}%")
    print(f"  Strict Pass (all-K): {report.pct_strict_pass * 100:.1f}% of scenarios")
    print(f"  Avg Consistency:     {report.avg_consistency:.3f}" if report.avg_consistency is not None else "  Avg Consistency:     n/a")
    print(f"  Total API calls:     {report.total_runs}")
    print(f"  Total cost:          ${report.total_cost_usd:.4f}")
    print(f"  Total time:          {report.total_time_ms / 1000:.1f}s")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass@K reliability runner — measures answer consistency across K runs"
    )
    parser.add_argument(
        "--connector", type=str, required=True,
        help="connector name (e.g. 'tmdb')"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="number of runs per scenario (default: 5)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="randomly sample this many scenarios (default: use all)"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="filter to a specific scenario category"
    )
    parser.add_argument(
        "--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD,
        dest="pass_threshold",
        help=f"min score to count a run as passing (default: {DEFAULT_PASS_THRESHOLD})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="path to write JSON results"
    )
    args = parser.parse_args()

    # load connector
    connector = get_connector(args.connector)
    logger.info(f"connector: {args.connector}")

    # load scenarios
    scenarios = load_scenarios(category=args.category)

    total_runs = (args.sample or len(scenarios)) * args.k
    logger.info(
        f"planning {args.sample or len(scenarios)} scenarios × {args.k} runs "
        f"= {total_runs} API calls"
    )

    report = run_reliability(
        scenarios=scenarios,
        connector=connector,
        k=args.k,
        pass_threshold=args.pass_threshold,
        sample=args.sample,
        seed=args.seed,
    )

    print_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_json())
        logger.info(f"results written to {out_path}")


if __name__ == "__main__":
    main()
