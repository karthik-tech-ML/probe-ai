"""
Run the full ProbeAI safety suite and produce a summary report.

Tests three categories across both RAG and agent pipelines:
  1. Prompt injection resistance (10 tests)
  2. Red team resistance (10 tests: 5 social engineering + 5 context override)
  3. Guardrail checks (topic boundary, output format, PII leakage)

Usage:
    python scripts/safety_report.py
    python scripts/safety_report.py --rag-only
    python scripts/safety_report.py --agent-only
    python scripts/safety_report.py --out safety_results.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.safety.injection import INJECTION_TESTS, score_injection_test
from src.safety.redteam import ALL_REDTEAM_TESTS, score_redteam_test
from src.safety.guardrails import run_all_guardrails


@dataclass
class SafetyResult:
    test_id: str
    test_type: str  # "injection", "redteam", or "guardrail"
    pipeline: str   # "rag" or "agent"
    question: str
    response: str
    scores: dict[str, Any]
    latency_ms: float


def _run_rag(question: str) -> tuple[str, float]:
    """Run a question through the RAG pipeline, return (answer, latency_ms)."""
    from src.rag.pipeline import ask
    result = ask(question)
    return result.answer, result.latency_ms


def _run_agent(question: str) -> tuple[str, float]:
    """Run a question through the agent pipeline, return (answer, latency_ms)."""
    from src.agent.graph import run_agent
    result = run_agent(question)
    return result.answer, result.latency_ms


def run_injection_tests(
    pipelines: list[str],
) -> list[SafetyResult]:
    """Run all 10 injection tests across specified pipelines."""
    results = []
    for test in INJECTION_TESTS:
        for pipeline in pipelines:
            logger.info(f"[{test.id}] ({pipeline}) {test.description}")
            try:
                if pipeline == "rag":
                    answer, latency = _run_rag(test.prompt)
                else:
                    answer, latency = _run_agent(test.prompt)

                scores = score_injection_test(test, answer)
                # also run guardrails on every response
                guardrail_scores = run_all_guardrails(answer, test.prompt)
                scores.update(guardrail_scores)

                results.append(SafetyResult(
                    test_id=test.id,
                    test_type="injection",
                    pipeline=pipeline,
                    question=test.prompt,
                    response=answer,
                    scores=scores,
                    latency_ms=latency,
                ))
            except Exception as e:
                logger.error(f"[{test.id}] ({pipeline}) failed: {e}")
                results.append(SafetyResult(
                    test_id=test.id,
                    test_type="injection",
                    pipeline=pipeline,
                    question=test.prompt,
                    response=f"ERROR: {e}",
                    scores={"injection_resisted": None, "error": str(e)},
                    latency_ms=0,
                ))
    return results


def run_redteam_tests(
    pipelines: list[str],
) -> list[SafetyResult]:
    """Run all 10 red team tests across specified pipelines."""
    results = []
    for test in ALL_REDTEAM_TESTS:
        for pipeline in pipelines:
            logger.info(f"[{test.id}] ({pipeline}) {test.description}")
            try:
                if pipeline == "rag":
                    answer, latency = _run_rag(test.prompt)
                else:
                    answer, latency = _run_agent(test.prompt)

                scores = score_redteam_test(test, answer)
                guardrail_scores = run_all_guardrails(answer, test.prompt)
                scores.update(guardrail_scores)

                results.append(SafetyResult(
                    test_id=test.id,
                    test_type="redteam",
                    pipeline=pipeline,
                    question=test.prompt,
                    response=answer,
                    scores=scores,
                    latency_ms=latency,
                ))
            except Exception as e:
                logger.error(f"[{test.id}] ({pipeline}) failed: {e}")
                results.append(SafetyResult(
                    test_id=test.id,
                    test_type="redteam",
                    pipeline=pipeline,
                    question=test.prompt,
                    response=f"ERROR: {e}",
                    scores={"redteam_resisted": None, "error": str(e)},
                    latency_ms=0,
                ))
    return results


def print_report(
    injection_results: list[SafetyResult],
    redteam_results: list[SafetyResult],
) -> None:
    """Print a clean summary report to stdout."""
    all_results = injection_results + redteam_results

    print(f"\n{'='*90}")
    print(f"  ProbeAI Safety Suite Report")
    print(f"  {len(all_results)} total tests across {_count_pipelines(all_results)} pipeline(s)")
    print(f"{'='*90}")

    # --- injection results ---
    print(f"\n  {'— Prompt Injection Resistance —':^86}")
    _print_injection_table(injection_results)

    # --- red team results ---
    print(f"\n  {'— Red Team Resistance —':^86}")
    _print_redteam_table(redteam_results)

    # --- guardrail summary ---
    print(f"\n  {'— Guardrail Pass Rates —':^86}")
    _print_guardrail_summary(all_results)

    # --- overall summary ---
    print(f"\n  {'— Overall Safety Summary —':^86}")
    _print_overall_summary(injection_results, redteam_results)
    print()


def _count_pipelines(results: list[SafetyResult]) -> int:
    return len(set(r.pipeline for r in results))


def _print_injection_table(results: list[SafetyResult]) -> None:
    header = f"  {'ID':<10} {'Pipeline':<8} {'Resisted':>9} {'Description':<50}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for r in results:
        resisted = r.scores.get("injection_resisted")
        status = "  ✓" if resisted == 1.0 else "  ✗" if resisted == 0.0 else "ERR"
        desc = r.scores.get("description", "")[:48]
        print(f"  {r.test_id:<10} {r.pipeline:<8} {status:>9} {desc:<50}")

    # subtotals per pipeline
    for pipeline in sorted(set(r.pipeline for r in results)):
        pipe_results = [r for r in results if r.pipeline == pipeline]
        resisted = sum(1 for r in pipe_results if r.scores.get("injection_resisted") == 1.0)
        total = sum(1 for r in pipe_results if r.scores.get("injection_resisted") is not None)
        rate = resisted / total if total > 0 else 0
        print(f"\n  {pipeline.upper()} injection resistance: {resisted}/{total} ({rate:.0%})")


def _print_redteam_table(results: list[SafetyResult]) -> None:
    header = f"  {'ID':<10} {'Pipeline':<8} {'Cat':<20} {'Resisted':>9} {'Description':<40}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for r in results:
        resisted = r.scores.get("redteam_resisted")
        status = "  ✓" if resisted == 1.0 else "  ✗" if resisted == 0.0 else "ERR"
        cat = r.scores.get("category", "")[:18]
        desc = r.scores.get("description", "")[:38]
        print(f"  {r.test_id:<10} {r.pipeline:<8} {cat:<20} {status:>9} {desc:<40}")

    # subtotals per pipeline
    for pipeline in sorted(set(r.pipeline for r in results)):
        pipe_results = [r for r in results if r.pipeline == pipeline]
        resisted = sum(1 for r in pipe_results if r.scores.get("redteam_resisted") == 1.0)
        total = sum(1 for r in pipe_results if r.scores.get("redteam_resisted") is not None)
        rate = resisted / total if total > 0 else 0
        print(f"\n  {pipeline.upper()} red team resistance: {resisted}/{total} ({rate:.0%})")


def _print_guardrail_summary(results: list[SafetyResult]) -> None:
    for pipeline in sorted(set(r.pipeline for r in results)):
        pipe_results = [r for r in results if r.pipeline == pipeline]
        valid = [r for r in pipe_results if "guardrails_all_pass" in r.scores]

        topic_pass = sum(1 for r in valid if r.scores.get("topic_boundary_pass"))
        format_pass = sum(1 for r in valid if r.scores.get("output_format_pass"))
        pii_pass = sum(1 for r in valid if r.scores.get("pii_leakage_pass"))
        total = len(valid)

        print(f"\n  {pipeline.upper()} pipeline ({total} responses checked):")
        print(f"    Topic boundary:  {topic_pass}/{total} pass")
        print(f"    Output format:   {format_pass}/{total} pass")
        print(f"    PII leakage:     {pii_pass}/{total} pass")


def _print_overall_summary(
    injection_results: list[SafetyResult],
    redteam_results: list[SafetyResult],
) -> None:
    all_results = injection_results + redteam_results

    for pipeline in sorted(set(r.pipeline for r in all_results)):
        pipe_inj = [r for r in injection_results if r.pipeline == pipeline]
        pipe_rt = [r for r in redteam_results if r.pipeline == pipeline]
        pipe_all = [r for r in all_results if r.pipeline == pipeline]

        inj_resisted = sum(1 for r in pipe_inj if r.scores.get("injection_resisted") == 1.0)
        inj_total = sum(1 for r in pipe_inj if r.scores.get("injection_resisted") is not None)

        rt_resisted = sum(1 for r in pipe_rt if r.scores.get("redteam_resisted") == 1.0)
        rt_total = sum(1 for r in pipe_rt if r.scores.get("redteam_resisted") is not None)

        valid_guardrail = [r for r in pipe_all if "guardrails_all_pass" in r.scores]
        guardrail_pass = sum(1 for r in valid_guardrail if r.scores.get("guardrails_all_pass"))
        guardrail_total = len(valid_guardrail)

        combined_resisted = inj_resisted + rt_resisted
        combined_total = inj_total + rt_total
        combined_rate = combined_resisted / combined_total if combined_total > 0 else 0

        print(f"\n  {pipeline.upper()} pipeline:")
        print(f"    Injection resistance:   {inj_resisted}/{inj_total}")
        print(f"    Red team resistance:    {rt_resisted}/{rt_total}")
        print(f"    Combined resistance:    {combined_resisted}/{combined_total} ({combined_rate:.0%})")
        print(f"    Guardrail pass rate:    {guardrail_pass}/{guardrail_total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ProbeAI safety suite")
    parser.add_argument("--rag-only", action="store_true",
                        help="only test the RAG pipeline")
    parser.add_argument("--agent-only", action="store_true",
                        help="only test the agent pipeline")
    parser.add_argument("--out", type=str, default=None,
                        help="path to write full JSON results")
    args = parser.parse_args()

    if args.rag_only:
        pipelines = ["rag"]
    elif args.agent_only:
        pipelines = ["agent"]
    else:
        pipelines = ["rag", "agent"]

    logger.info(f"running safety suite on pipelines: {pipelines}")

    t0 = time.perf_counter()

    injection_results = run_injection_tests(pipelines)
    redteam_results = run_redteam_tests(pipelines)

    total_time = time.perf_counter() - t0

    print_report(injection_results, redteam_results)
    print(f"  Total time: {total_time:.1f}s")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "injection_results": [
                {"test_id": r.test_id, "pipeline": r.pipeline,
                 "response": r.response, "scores": r.scores, "latency_ms": r.latency_ms}
                for r in injection_results
            ],
            "redteam_results": [
                {"test_id": r.test_id, "pipeline": r.pipeline,
                 "response": r.response, "scores": r.scores, "latency_ms": r.latency_ms}
                for r in redteam_results
            ],
        }
        out_path.write_text(json.dumps(output, indent=2))
        logger.info(f"results written to {out_path}")


if __name__ == "__main__":
    main()
