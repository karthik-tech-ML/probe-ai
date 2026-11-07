"""
Trace for the multi-agent system: which agent did what, in what order.

MultiAgentTrace subclasses AgentTrace, so everything that reads an
AgentTrace (tool_selection, task_success, the eval runner) works on it
unchanged. On top of the flat tool-call list it keeps an ordered list of
steps, each tagged with the agent that took it, so the multi-agent
metrics can reason about routing, handoffs, loops and the verifier.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.memory import AgentTrace, _summarize_output

STEP_TYPES = ("route", "tool_call", "handoff", "draft", "verify", "final")

SUPERVISOR = "supervisor"
VERIFIER = "verifier_agent"
SPECIALISTS = ("retrieval_agent", "structured_agent", "analyst_agent")

# longest input/output string kept per step in to_dict()
_MAX_TEXT = 500


@dataclass
class TraceStep:
    """One thing one agent did."""

    agent_name: str
    step_type: str  # one of STEP_TYPES
    input: Any = None
    output: Any = None
    # for route/handoff: the agent being routed to
    target: str | None = None
    # LLM calls and tokens spent on this step (0 for pure bookkeeping steps)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.step_type not in STEP_TYPES:
            raise ValueError(
                f"unknown step_type {self.step_type!r}, expected one of {STEP_TYPES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "type": self.step_type,
            "target": self.target,
            "input": _clip(self.input),
            "output": _clip(self.output),
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "timestamp": round(self.timestamp, 3),
        }


def _clip(value: Any) -> Any:
    """Keep step payloads readable: summarize lists, truncate long strings."""
    if isinstance(value, list):
        return _summarize_output(value)
    text = value if isinstance(value, str) else repr(value) if value is not None else None
    if text is not None and len(text) > _MAX_TEXT:
        return text[:_MAX_TEXT] + "..."
    return value if isinstance(value, (str, int, float, bool)) or value is None else text


@dataclass
class MultiAgentTrace(AgentTrace):
    """AgentTrace plus a per-agent step log."""

    steps: list[TraceStep] = field(default_factory=list)
    # set when the supervisor hit the step cap and we returned a best-effort answer
    step_cap_hit: bool = False

    def add_step(
        self,
        agent_name: str,
        step_type: str,
        input: Any = None,
        output: Any = None,
        *,
        target: str | None = None,
        llm_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> TraceStep:
        """Record a step and roll its tokens into the run totals."""
        step = TraceStep(
            agent_name=agent_name,
            step_type=step_type,
            input=input,
            output=output,
            target=target,
            llm_calls=llm_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.steps.append(step)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.llm_calls += llm_calls
        return step

    def add_agent_tool_call(
        self, agent_name: str, name: str, input: dict, output: Any
    ) -> None:
        """A tool call made by a specific agent. Also feeds the flat tool_calls list."""
        self.add_tool_call(name=name, input=input, output=output)
        self.add_step(agent_name, "tool_call", input={"tool": name, **input}, output=output)

    # -- aggregate views used by the multi-agent metrics -------------------

    @property
    def agents_used(self) -> list[str]:
        """Distinct agents that took at least one step, in first-appearance order."""
        seen: list[str] = []
        for s in self.steps:
            if s.agent_name not in seen:
                seen.append(s.agent_name)
        return seen

    @property
    def route_sequence(self) -> list[str]:
        """Agents the supervisor routed to, in order (repeats kept)."""
        return [s.target for s in self.steps if s.step_type == "route" and s.target]

    @property
    def handoff_count(self) -> int:
        return sum(1 for s in self.steps if s.step_type == "handoff")

    @property
    def llm_call_count(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    @property
    def verifier_actions(self) -> list[str]:
        """'approve' / 'revise' for each verify step, in order."""
        actions = []
        for s in self.steps_of("verify"):
            word = str(s.output or "").split(":", 1)[0].strip().lower()
            if word in ("approve", "revise"):
                actions.append(word)
        return actions

    def steps_of(self, step_type: str) -> list[TraceStep]:
        return [s for s in self.steps if s.step_type == step_type]

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "agents_used": self.agents_used,
            "route_sequence": self.route_sequence,
            "handoff_count": self.handoff_count,
            "llm_call_count": self.llm_call_count,
            "verifier_actions": self.verifier_actions,
            "step_cap_hit": self.step_cap_hit,
            "steps": [s.to_dict() for s in self.steps],
        })
        return d
