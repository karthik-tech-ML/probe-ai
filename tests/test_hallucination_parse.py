"""Regression: the hallucination judge sometimes wraps its JSON in prose."""

from src.evaluation.metrics.hallucination import _parse_response

BODY = '{"hallucination_score": 0.0, "claims": []}'


def test_bare_json():
    assert _parse_response(BODY, "t")["hallucination_score"] == 0.0


def test_fenced_json():
    assert _parse_response(f"```json\n{BODY}\n```", "t") is not None


def test_prose_around_json():
    raw = f"Here is my analysis of the claims:\n\n{BODY}\n\nLet me know if you need more detail."
    assert _parse_response(raw, "t")["claims"] == []


def test_garbage_returns_none():
    assert _parse_response("no json here at all", "t") is None
