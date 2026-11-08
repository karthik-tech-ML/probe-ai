"""Run-level caps: multi-agent token budget and single-agent tool-round limit. No API calls."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from src.agent.graph import MAX_TOOL_ROUNDS, run_with_round_limit
from src.multi_agent.graph import MAX_RUN_TOKENS
from src.multi_agent.trace import MultiAgentTrace


class _LoopingAgent:
    """Streams a few states, then raises as LangGraph does at the recursion limit."""

    def stream(self, inputs, config=None, stream_mode=None):
        assert config["recursion_limit"] == 2 * MAX_TOOL_ROUNDS + 1
        yield {"messages": [HumanMessage("q")]}
        yield {"messages": [HumanMessage("q"), AIMessage("partial")]}
        raise GraphRecursionError("limit")


class _FinishingAgent:
    def stream(self, inputs, config=None, stream_mode=None):
        yield {"messages": [AIMessage("done")]}


def test_single_agent_limit_keeps_last_state_and_flags_it():
    state, hit = run_with_round_limit(_LoopingAgent(), "q")
    assert hit is True
    assert state["messages"][-1].content == "partial"


def test_single_agent_normal_run_not_flagged():
    state, hit = run_with_round_limit(_FinishingAgent(), "q")
    assert hit is False and state["messages"][-1].content == "done"


def test_trace_records_token_budget_flag():
    t = MultiAgentTrace(question="q")
    assert t.token_budget_hit is False
    t.add_step("supervisor", "route", "x", "y", input_tokens=MAX_RUN_TOKENS, output_tokens=1)
    assert t.total_input_tokens + t.total_output_tokens >= MAX_RUN_TOKENS
    t.token_budget_hit = True
    assert t.to_dict()["token_budget_hit"] is True
