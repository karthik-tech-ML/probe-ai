"""Tests for MultiAgentTrace. Hand-built traces only, no API calls."""

import pytest

from src.agent.memory import AgentTrace
from src.multi_agent.trace import MultiAgentTrace, TraceStep


def _sample_trace() -> MultiAgentTrace:
    t = MultiAgentTrace(question="Which Nolan film earned the most?")
    t.add_step("supervisor", "route", "plan", "structured first", target="structured_agent",
               llm_calls=1, input_tokens=300, output_tokens=40)
    t.add_step("supervisor", "handoff", target="structured_agent")
    t.add_agent_tool_call("structured_agent", "search_by_director",
                          {"director_name": "Christopher Nolan"}, [{"movie_id": 27205}])
    t.add_step("structured_agent", "draft", "results", "Inception",
               llm_calls=2, input_tokens=900, output_tokens=60)
    t.add_step("supervisor", "route", "next", "analyze", target="analyst_agent",
               llm_calls=1, input_tokens=350, output_tokens=30)
    t.add_step("supervisor", "handoff", target="analyst_agent")
    t.add_step("analyst_agent", "draft", "compare", "Inception earned most",
               llm_calls=1, input_tokens=500, output_tokens=80)
    t.add_step("verifier_agent", "verify", "draft", "approve",
               llm_calls=1, input_tokens=600, output_tokens=20)
    t.add_step("supervisor", "final", output="Inception",
               llm_calls=1, input_tokens=400, output_tokens=25)
    t.final_answer = "Inception"
    return t


def test_is_an_agent_trace():
    assert isinstance(_sample_trace(), AgentTrace)


def test_existing_properties_still_work():
    t = _sample_trace()
    assert t.tools_used == ["search_by_director"]
    assert t.num_tool_calls == 1
    assert t.tool_calls[0].tool_name == "search_by_director"
    d = t.to_dict()
    assert d["num_tool_calls"] == 1
    assert d["tools_used"] == ["search_by_director"]
    assert d["final_answer"] == "Inception"


def test_agents_used_is_ordered_and_distinct():
    assert _sample_trace().agents_used == [
        "supervisor", "structured_agent", "analyst_agent", "verifier_agent"
    ]


def test_route_sequence_and_handoffs():
    t = _sample_trace()
    assert t.route_sequence == ["structured_agent", "analyst_agent"]
    assert t.handoff_count == 2


def test_llm_call_count_and_tokens_roll_up():
    t = _sample_trace()
    assert t.llm_call_count == 7
    assert t.total_input_tokens == 300 + 900 + 350 + 500 + 600 + 400
    assert t.total_output_tokens == 40 + 60 + 30 + 80 + 20 + 25


def test_repeated_routes_are_kept():
    t = MultiAgentTrace(question="q")
    for _ in range(2):
        t.add_step("supervisor", "route", target="retrieval_agent")
    assert t.route_sequence == ["retrieval_agent", "retrieval_agent"]


def test_to_dict_includes_steps_and_summary_fields():
    d = _sample_trace().to_dict()
    assert d["agents_used"][0] == "supervisor"
    assert d["llm_call_count"] == 7
    assert d["handoff_count"] == 2
    assert d["step_cap_hit"] is False
    assert len(d["steps"]) == 9
    assert {"agent", "type", "input", "output", "timestamp"} <= set(d["steps"][0])


def test_long_payloads_are_truncated_in_to_dict():
    t = MultiAgentTrace(question="q")
    t.add_step("analyst_agent", "draft", output="x" * 5000)
    out = t.to_dict()["steps"][0]["output"]
    assert len(out) < 600 and out.endswith("...")


def test_empty_trace_is_safe():
    t = MultiAgentTrace(question="q")
    assert t.agents_used == [] and t.route_sequence == []
    assert t.handoff_count == 0 and t.llm_call_count == 0


def test_unknown_step_type_rejected():
    with pytest.raises(ValueError):
        TraceStep(agent_name="supervisor", step_type="dance")
