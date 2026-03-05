"""
Smoke test for the LangGraph agent.

Runs 3 multi-step questions that single-shot RAG can't handle well,
shows the tool call sequence and final answer for each.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import run_agent

QUESTIONS = [
    "What Christopher Nolan movies are in the science fiction genre?",
    "Which actors appeared in both a Christopher Nolan film and a Quentin Tarantino film?",
    "What is the highest-grossing action movie in the dataset?",
]


def main() -> None:
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"  Test {i}: {question}")
        print(f"{'='*70}")

        result = run_agent(question)

        print(f"\n  Tool call sequence:")
        for j, tc in enumerate(result.trace.tool_calls, 1):
            # show a compact summary of what went in and came out
            input_str = ", ".join(f"{k}={v!r}" for k, v in tc.tool_input.items())
            if isinstance(tc.tool_output, str):
                # tool output is serialized — count results
                out_str = f"{len(tc.tool_output)} chars"
            elif isinstance(tc.tool_output, list):
                out_str = f"{len(tc.tool_output)} results"
            else:
                out_str = str(tc.tool_output)[:80]
            print(f"    {j}. {tc.tool_name}({input_str}) → {out_str}")

        print(f"\n  Final answer:\n    {result.answer}")
        print(f"\n  Stats:")
        print(f"    Tool calls: {result.trace.num_tool_calls}")
        print(f"    Tools used: {result.trace.tools_used}")
        print(f"    Tokens: {result.trace.total_input_tokens}in / {result.trace.total_output_tokens}out")
        print(f"    Latency: {result.latency_ms:.0f}ms")


if __name__ == "__main__":
    main()
