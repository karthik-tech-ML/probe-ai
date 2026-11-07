"""Tests for the per-turn tool-round cap. A fake agent stands in for the ReAct loop: no API."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from src.multi_agent.graph import MAX_TOOL_ROUNDS, run_react_with_limit


class FakeAgent:
    """Yields `n_states` states, then raises GraphRecursionError if `loops_forever`."""

    def __init__(self, n_states: int, loops_forever: bool):
        self.n_states, self.loops_forever = n_states, loops_forever
        self.config = None

    def stream(self, inputs, config=None, stream_mode=None):
        self.config, self.stream_mode = config, stream_mode
        for i in range(self.n_states):
            yield {"messages": inputs["messages"] + [AIMessage(content=f"state {i}")]}
        if self.loops_forever:
            raise GraphRecursionError("recursion limit hit")


def test_normal_run_is_not_flagged():
    agent = FakeAgent(3, loops_forever=False)
    state, hit = run_react_with_limit(agent, "find films")
    assert hit is False
    assert state["messages"][-1].content == "state 2"


def test_runaway_keeps_last_state_and_flags_it():
    agent = FakeAgent(4, loops_forever=True)
    state, hit = run_react_with_limit(agent, "find films")
    assert hit is True
    assert state["messages"][-1].content == "state 3"  # progress is kept, not lost


def test_recursion_limit_matches_round_budget():
    agent = FakeAgent(1, loops_forever=False)
    run_react_with_limit(agent, "q", max_rounds=4)
    assert agent.config == {"recursion_limit": 9}  # 2 * 4 rounds + 1 final reply
    assert agent.stream_mode == "values"


def test_default_budget_is_bounded():
    agent = FakeAgent(1, loops_forever=False)
    run_react_with_limit(agent, "q")
    assert agent.config["recursion_limit"] == 2 * MAX_TOOL_ROUNDS + 1


def test_failure_before_any_state_returns_the_task_message():
    agent = FakeAgent(0, loops_forever=True)
    state, hit = run_react_with_limit(agent, "find films")
    assert hit is True
    assert isinstance(state["messages"][0], HumanMessage)
