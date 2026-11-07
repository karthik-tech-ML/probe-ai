"""Tests for the mechanical multi-agent metrics. Hand-built traces, no API calls."""

from types import SimpleNamespace

from src.evaluation.metrics.coordination import score_coordination
from src.evaluation.metrics.cost import score_multi_agent_cost
from src.evaluation.metrics.routing import collapse_repeats, score_routing
from src.evaluation.metrics.verifier import score_verifier
from src.evaluation.runner import Scenario
from src.multi_agent.trace import MultiAgentTrace


def _scenario(expected_route=None) -> Scenario:
    return Scenario(
        id="sc_t", question="q", expected_answer="a", category="multi_hop",
        expected_source_ids=[], difficulty="hard", hallucination_risk="low",
        expected_route=expected_route or [],
    )


def _result(trace=None, scores=None, latency_ms=1234.0):
    r = SimpleNamespace(latency_ms=latency_ms)
    if trace is not None:
        r._agent_trace = trace
    if scores is not None:
        r._scores = scores
    return r


def _route(trace, *targets, task="t"):
    for i, t in enumerate(targets):
        trace.add_step("supervisor", "route", target=t, llm_calls=1, input_tokens=100, output_tokens=10)
        trace.add_step("supervisor", "handoff", f"{task}{i}", target=t)


# -- routing ---------------------------------------------------------------

def test_collapse_repeats():
    assert collapse_repeats(["a", "a", "b", "a"]) == ["a", "b", "a"]
    assert collapse_repeats([]) == []


def test_routing_exact_match():
    t = MultiAgentTrace(question="q")
    _route(t, "structured_agent", "analyst_agent")
    out = score_routing(_scenario(["structured_agent", "analyst_agent"]), _result(t))
    assert out["routing_accuracy"] == 1.0
    assert out["routing_exact_order"] is True


def test_routing_repeats_collapse_before_comparison():
    t = MultiAgentTrace(question="q")
    _route(t, "structured_agent", "structured_agent", "analyst_agent")
    out = score_routing(_scenario(["structured_agent", "analyst_agent"]), _result(t))
    assert out["routing_exact_order"] is True


def test_routing_right_agents_wrong_order():
    t = MultiAgentTrace(question="q")
    _route(t, "analyst_agent", "structured_agent")
    out = score_routing(_scenario(["structured_agent", "analyst_agent"]), _result(t))
    assert out["routing_precision"] == 1.0 and out["routing_recall"] == 1.0
    assert out["routing_exact_order"] is False


def test_routing_extra_and_missing_specialists():
    t = MultiAgentTrace(question="q")
    _route(t, "retrieval_agent", "structured_agent")
    out = score_routing(_scenario(["structured_agent", "analyst_agent"]), _result(t))
    assert out["routing_precision"] == 0.5
    assert out["routing_recall"] == 0.5
    assert out["routing_accuracy"] == 0.5


def test_routing_nothing_routed_scores_zero():
    out = score_routing(_scenario(["structured_agent"]), _result(MultiAgentTrace(question="q")))
    assert out["routing_precision"] == 0.0 and out["routing_recall"] == 0.0
    assert out["routing_accuracy"] == 0.0


def test_routing_none_without_expected_route_or_trace():
    t = MultiAgentTrace(question="q")
    assert score_routing(_scenario([]), _result(t))["routing_accuracy"] is None
    out = score_routing(_scenario(["structured_agent"]), _result(None))
    assert out["routing_accuracy"] is None and "no multi-agent trace" in out["routing_reason"]


# -- coordination ----------------------------------------------------------

def test_coordination_counts_handoffs_and_steps():
    t = MultiAgentTrace(question="q")
    _route(t, "structured_agent")
    t.add_step("structured_agent", "handoff", "result", target="supervisor")
    t.add_step("supervisor", "final", output="done")
    out = score_coordination(_scenario(), _result(t))
    assert out["handoff_count"] == 2
    assert out["loop_detected"] is False
    # route, handoff out, handoff back; the final step is not counted
    assert out["steps_to_answer"] == 3


def test_coordination_detects_repeated_call():
    t = MultiAgentTrace(question="q")
    for _ in range(2):
        t.add_step("supervisor", "handoff", "  Find Nolan   films ", target="structured_agent")
    out = score_coordination(_scenario(), _result(t))
    assert out["loop_detected"] is True and out["loop_reason"] == "repeated_call"


def test_coordination_same_agent_different_task_is_not_a_loop():
    t = MultiAgentTrace(question="q")
    t.add_step("supervisor", "handoff", "find Nolan films", target="structured_agent")
    t.add_step("supervisor", "handoff", "find Tarantino films", target="structured_agent")
    assert score_coordination(_scenario(), _result(t))["loop_detected"] is False


def test_coordination_step_cap_counts_as_loop():
    t = MultiAgentTrace(question="q")
    t.step_cap_hit = True
    out = score_coordination(_scenario(), _result(t))
    assert out["loop_detected"] is True and out["loop_reason"] == "step_cap"


def test_coordination_none_without_trace():
    assert score_coordination(_scenario(), _result(None))["handoff_count"] is None


# -- cost ------------------------------------------------------------------

def test_cost_reports_calls_tokens_latency():
    t = MultiAgentTrace(question="q")
    _route(t, "structured_agent", "analyst_agent")  # 2 LLM calls, 200in / 20out
    out = score_multi_agent_cost(_scenario(), _result(t, scores={"task_success_score": 1.0}))
    assert out["llm_call_count"] == 2
    assert out["ma_total_tokens"] == 220
    assert out["ma_latency_ms"] == 1234.0
    assert out["tokens_per_correct_answer"] == 220


def test_tokens_per_correct_answer_none_when_not_fully_correct():
    t = MultiAgentTrace(question="q")
    _route(t, "structured_agent")
    for ts in (0.5, 0.0, None):
        out = score_multi_agent_cost(_scenario(), _result(t, scores={"task_success_score": ts}))
        assert out["tokens_per_correct_answer"] is None
    assert score_multi_agent_cost(_scenario(), _result(t))["tokens_per_correct_answer"] is None


def test_cost_none_without_trace():
    assert score_multi_agent_cost(_scenario(), _result(None))["llm_call_count"] is None


# -- verifier --------------------------------------------------------------

def test_verifier_approve():
    t = MultiAgentTrace(question="q")
    t.add_step("verifier_agent", "verify", "draft A", "approve: all supported")
    t.final_answer = "draft A"
    out = score_verifier(_scenario(), _result(t))
    assert out["verifier_action"] == "approve"
    assert out["verifier_revision_changed_answer"] is None


def test_verifier_revise_that_changed_the_answer():
    t = MultiAgentTrace(question="q")
    t.add_step("verifier_agent", "verify", "draft A", "revise: claim unsupported")
    t.final_answer = "draft B"
    out = score_verifier(_scenario(), _result(t))
    assert out["verifier_action"] == "revise"
    assert out["verifier_revision_changed_answer"] is True


def test_verifier_revise_that_changed_nothing():
    t = MultiAgentTrace(question="q")
    t.add_step("verifier_agent", "verify", "Draft  A", "revise: meh")
    t.final_answer = "draft a"
    assert score_verifier(_scenario(), _result(t))["verifier_revision_changed_answer"] is False


def test_verifier_did_not_run():
    t = MultiAgentTrace(question="q")
    t.step_cap_hit = True
    out = score_verifier(_scenario(), _result(t))
    assert out["verifier_action"] is None and "step cap" in out["verifier_reason"]


def test_verifier_none_without_trace():
    assert score_verifier(_scenario(), _result(None))["verifier_action"] is None
