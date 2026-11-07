"""Tests for the attribution judge. The API call is patched out: no network."""

from types import SimpleNamespace

import pytest

from src.evaluation.metrics import attribution as attr
from src.evaluation.runner import Scenario
from src.multi_agent.trace import MultiAgentTrace

GOOD = '{"culprit_agent": "structured_agent", "failure_type": "bad_tool_args", ' \
       '"reason": "Searched the wrong director.", "error_propagated": true}'


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(attr, "_CACHE_PATH", tmp_path / "attribution.json")
    monkeypatch.setattr(attr.settings, "anthropic_api_key", "test-key")
    calls = []

    def fake_judge(prompt):
        calls.append(prompt)
        return fake_judge.reply

    fake_judge.reply = GOOD
    monkeypatch.setattr(attr, "_call_judge", fake_judge)
    return calls


def _scenario():
    return Scenario(id="sc_t", question="q", expected_answer="Inception", category="multi_hop",
                    expected_source_ids=[], difficulty="hard", hallucination_risk="low")


def _trace(answer="Interstellar"):
    t = MultiAgentTrace(question="q")
    t.add_step("supervisor", "route", "x", "y", target="structured_agent")
    t.add_agent_tool_call("structured_agent", "search_by_director", {"director_name": "Nolan"}, "[...]")
    t.final_answer = answer
    return t


def _result(trace, scores):
    return SimpleNamespace(_agent_trace=trace, _scores=scores, latency_ms=1.0)


def test_wrong_gate():
    assert attr.answer_is_wrong({"task_success_score": 0.5}) is True
    assert attr.answer_is_wrong({"task_success_score": 1.0, "hallucination_score": 0.2}) is True
    assert attr.answer_is_wrong({"task_success_score": 1.0, "hallucination_score": 0.0}) is False
    assert attr.answer_is_wrong({"task_success_score": None}) is None
    assert attr.answer_is_wrong({}) is None


def test_correct_answer_makes_no_call(_isolate):
    out = attr.score_attribution(_scenario(), _result(_trace(), {"task_success_score": 1.0, "hallucination_score": 0.0}))
    assert out["attribution_culprit_agent"] is None and "not wrong" in out["attribution_reason"]
    assert _isolate == []


def test_no_trace_and_no_scores_make_no_call(_isolate):
    assert attr.score_attribution(_scenario(), SimpleNamespace(latency_ms=1))["attribution_culprit_agent"] is None
    out = attr.score_attribution(_scenario(), _result(_trace(), {}))
    assert out["attribution_culprit_agent"] is None
    assert _isolate == []


def test_wrong_answer_gets_verdict(_isolate):
    out = attr.score_attribution(_scenario(), _result(_trace(), {"task_success_score": 0.0}))
    assert out["attribution_culprit_agent"] == "structured_agent"
    assert out["attribution_failure_type"] == "bad_tool_args"
    assert out["attribution_error_propagated"] is True
    assert out["attribution_cached"] is False
    assert len(_isolate) == 1 and "search_by_director" in _isolate[0]


def test_second_identical_run_hits_cache(_isolate):
    r = _result(_trace(), {"task_success_score": 0.0})
    attr.score_attribution(_scenario(), r)
    out = attr.score_attribution(_scenario(), r)
    assert out["attribution_cached"] is True
    assert len(_isolate) == 1


def test_changed_trace_misses_cache(_isolate):
    attr.score_attribution(_scenario(), _result(_trace("A"), {"task_success_score": 0.0}))
    attr.score_attribution(_scenario(), _result(_trace("B"), {"task_success_score": 0.0}))
    assert len(_isolate) == 2


def test_invalid_labels_are_flagged_not_cached(_isolate, monkeypatch):
    attr._call_judge.reply = '{"culprit_agent": "the_butler", "failure_type": "vibes", "reason": "r", "error_propagated": "yes"}'
    r = _result(_trace(), {"task_success_score": 0.0})
    out = attr.score_attribution(_scenario(), r)
    assert out["attribution_culprit_agent"] is None
    assert out["attribution_failure_type"] is None
    assert out["attribution_error_propagated"] is None
    assert len(out["attribution_invalid_fields"]) == 2
    attr.score_attribution(_scenario(), r)
    assert len(_isolate) == 2  # not cached, so the judge was asked again


def test_unparseable_reply(_isolate):
    attr._call_judge.reply = "I think the supervisor did it."
    out = attr.score_attribution(_scenario(), _result(_trace(), {"task_success_score": 0.0}))
    assert out["attribution_culprit_agent"] is None and "unparseable" in out["attribution_reason"]


def test_fenced_json_is_accepted(_isolate):
    attr._call_judge.reply = "```json\n" + GOOD + "\n```"
    out = attr.score_attribution(_scenario(), _result(_trace(), {"task_success_score": 0.0}))
    assert out["attribution_culprit_agent"] == "structured_agent"
