"""
Conversation state tracking for the agent.

Keeps a record of which tools were called, what they returned,
and the final answer — all within a single question. This isn't
multi-turn chat memory; it's a trace of one agent run so the
eval harness can see exactly what happened.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any


@dataclass
class AgentTrace:
    """Everything that happened during one agent run."""

    question: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    # total token usage across all LLM calls in this run
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_tool_call(self, name: str, input: dict, output: Any) -> None:
        self.tool_calls.append(ToolCall(
            tool_name=name,
            tool_input=input,
            tool_output=output,
        ))

    @property
    def tools_used(self) -> list[str]:
        return [tc.tool_name for tc in self.tool_calls]

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "num_tool_calls": self.num_tool_calls,
            "tools_used": self.tools_used,
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "input": tc.tool_input,
                    # truncate big outputs so the trace is readable
                    "output_summary": _summarize_output(tc.tool_output),
                }
                for tc in self.tool_calls
            ],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


def _summarize_output(output: Any, max_items: int = 10) -> Any:
    """Keep tool output summaries short for trace readability."""
    if isinstance(output, list):
        count = len(output)
        items = output[:max_items]
        # for movie results, just show title + id
        summarized = []
        for item in items:
            if isinstance(item, dict) and "title" in item:
                summarized.append({
                    k: v for k, v in item.items()
                    if k in ("movie_id", "title", "director", "genres", "revenue", "rating")
                })
            else:
                summarized.append(item)
        if count > max_items:
            summarized.append(f"... and {count - max_items} more")
        return summarized
    return output
