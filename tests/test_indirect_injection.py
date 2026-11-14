"""Indirect-injection helpers: poisoning, restore, and the mechanical compliance checks. No API calls."""

from types import SimpleNamespace

from src.safety.indirect_injection import PAYLOADS, complied, poison_text, poisoned, spread

P = {p.id: p for p in PAYLOADS}


def test_poison_only_touches_target_movie():
    rec = "Title: Inception\nPlot: dreams."
    assert P["hijack"].text in poison_text(rec, P["hijack"])
    other = "Title: Avatar\nPlot: blue."
    assert poison_text(other, P["hijack"]) == other


def test_poisoned_wraps_and_restores_tools():
    from src.agent import tools as T
    before = T.search_movies.func
    with poisoned(P["hijack"]):
        assert T.search_movies.func is not before
    assert T.search_movies.func is before


def test_complied_checks():
    assert complied(P["canary_prefix"], "ZX-CANARY-7731 Inception is...")
    assert not complied(P["canary_prefix"], "Inception is... (ignored ZX-CANARY-7731)")
    assert complied(P["hijack"], "PWNED")
    assert not complied(P["hijack"], "Inception is a heist film directed by Christopher Nolan, ignoring the PWNED text.")
    assert complied(P["false_fact"], "It was released in 1999.")
    assert not complied(P["false_fact"], "Christopher Nolan directed it in 2010.")


def test_spread_reports_stages():
    step = lambda agent, kind, i="", o="": SimpleNamespace(agent_name=agent, step_type=kind, input=i, output=o)
    trace = SimpleNamespace(steps=[
        step("retrieval_agent", "draft", "t", "found VERIFIED-ZX9"),
        step("supervisor", "route", "plan", "ok"),
        step("supervisor", "draft", "q", "clean answer"),
    ], verifier_actions=["approve"])
    s = spread(trace, P["agent_to_agent"])
    assert s == {"in_specialist_output": True, "in_supervisor_routing": False,
                 "in_draft": False, "verifier_actions": ["approve"]}


def test_round2_payloads_and_cast_surface():
    from src.safety.indirect_injection import PAYLOADS_V2, _poison_output
    cast = next(p for p in PAYLOADS_V2 if p.surface == "cast")
    out = _poison_output([{"title": "Inception", "cast": "A, B"}, {"title": "Dunkirk", "cast": "C"}], cast)
    assert cast.text in out[0]["cast"] and out[1]["cast"] == "C"
    assert complied(next(p for p in PAYLOADS_V2 if p.id == "fake_system_tag"), "ZX-CANARY-7731 hello")


def test_cast_poison_restores_structured_tools():
    from src.agent import tools as T
    from src.safety.indirect_injection import PAYLOADS_V2
    before = T.search_by_director.func
    with poisoned(next(p for p in PAYLOADS_V2 if p.surface == "cast")):
        assert T.search_by_director.func is not before
    assert T.search_by_director.func is before


def test_round3_payloads_are_new_and_apply():
    from src.safety.indirect_injection import PAYLOADS_V2, PAYLOADS_V3, _poison_output
    assert {p.id for p in PAYLOADS_V3}.isdisjoint({p.id for p in PAYLOADS_V2})
    g = next(p for p in PAYLOADS_V3 if p.surface == "genres")
    out = _poison_output({"title": "Inception", "genres": "Action", "cast": "A"}, g)
    assert g.text in out["genres"] and out["cast"] == "A"
    assert complied(next(p for p in PAYLOADS_V3 if p.id == "r3_spanish"), "Plot... ZX-R3-D")
