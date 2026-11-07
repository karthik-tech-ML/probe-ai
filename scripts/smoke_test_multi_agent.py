"""
Smoke test for the multi-agent system.

Runs one multi_field, one aggregation and one multi_hop scenario from the
library through the supervisor + specialists system and prints the route,
handoffs, LLM calls, tokens, latency and the verifier's verdict for each.

    python scripts/smoke_test_multi_agent.py
    python scripts/smoke_test_multi_agent.py --trace   # also dump every step
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.runner import load_scenarios
from src.multi_agent.graph import run_multi_agent

CATEGORIES = ["multi_field", "aggregation", "multi_hop"]


def _verifier_summary(trace) -> str:
    verdicts = [s.output for s in trace.steps_of("verify")]
    if not verdicts:
        return "not run" + (" (step cap hit)" if trace.step_cap_hit else "")
    return " -> ".join(str(v).split(":")[0] for v in verdicts) + (
        f"  ({verdicts[0]})" if str(verdicts[0]).startswith("revise") else ""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="print every trace step")
    args = parser.parse_args()

    scenarios = [load_scenarios(category=c)[0] for c in CATEGORIES]

    for sc in scenarios:
        print(f"\n{'=' * 78}\n  [{sc.id}] ({sc.category}) {sc.question}\n  expected: {sc.expected_answer}\n{'=' * 78}")
        result = run_multi_agent(sc.question)
        t = result.trace

        print(f"\n  Answer:\n    {result.answer}")
        print(f"\n  Route sequence:  {' -> '.join(t.route_sequence) or '(none)'}")
        print(f"  Agents used:     {', '.join(t.agents_used)}")
        tool_steps = [f"{s.agent_name}:{s.input.get('tool')}" for s in t.steps_of("tool_call")]
        print(f"  Tool calls:      {tool_steps}")
        print(f"  Handoffs:        {t.handoff_count}")
        print(f"  LLM calls:       {t.llm_call_count}")
        print(f"  Tokens:          {t.total_input_tokens}in / {t.total_output_tokens}out")
        print(f"  Latency:         {result.latency_ms:.0f}ms")
        print(f"  Verifier:        {_verifier_summary(t)}")
        print(f"  Step cap hit:    {t.step_cap_hit}")

        if args.trace:
            print("\n  Trace:")
            for i, s in enumerate(t.steps, 1):
                tgt = f" -> {s.target}" if s.target else ""
                print(f"    {i:>2}. {s.agent_name:<17} {s.step_type:<9}{tgt}  {str(s.output)[:90]!r}")


if __name__ == "__main__":
    main()
