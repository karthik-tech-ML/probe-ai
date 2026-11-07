"""
Supervisor + specialists, built on a LangGraph StateGraph.

    supervisor -> {retrieval_agent | structured_agent | analyst_agent} -> supervisor ...
    supervisor -> draft -> verifier_agent -> (revise: supervisor | approve: final)

The supervisor makes a real routing decision each turn (one specialist or
finish). Specialists reuse the tools from src.agent.tools. Every step goes
into a MultiAgentTrace. Total node executions are capped; on hitting the
cap the supervisor is forced to finish and the verifier is skipped.
"""

import time
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger

from src.agent.tools import (
    filter_movies,
    get_movie_details,
    search_by_cast,
    search_by_director,
    search_by_genre,
    search_movies,
)
from src.config import settings
from src.multi_agent import prompts
from src.multi_agent.trace import SPECIALISTS, SUPERVISOR, VERIFIER, MultiAgentTrace
from src.multi_agent.utils import parse_json_object

MAX_STEPS = 12
# one verifier-triggered revision at most
MAX_REVISIONS = 1
# per tool output kept in evidence, same bound the eval runner uses for context
_MAX_EVIDENCE_CHARS = 3000

_TOOLS_BY_AGENT = {
    "retrieval_agent": [search_movies, get_movie_details],
    "structured_agent": [search_by_director, search_by_cast, search_by_genre, filter_movies],
}
_SYSTEM_BY_AGENT = {
    "retrieval_agent": prompts.RETRIEVAL_SYSTEM,
    "structured_agent": prompts.STRUCTURED_SYSTEM,
}


class MultiAgentState(TypedDict):
    question: str
    plan: list[str]
    evidence: list[dict[str, Any]]
    next_agent: str
    task: str
    draft: str
    verifier_feedback: str
    revise_count: int
    steps_taken: int
    cap_hit: bool
    answer: str


@dataclass
class MultiAgentResult:
    question: str
    answer: str
    trace: MultiAgentTrace
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "latency_ms": round(self.latency_ms, 1),
            "trace": self.trace.to_dict(),
        }


# -- helpers ----------------------------------------------------------------

def _llm(max_tokens: int = 1024) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=max_tokens,
        temperature=0,
    )


def _text(content: Any) -> str:
    """Message content is a str, or a list of blocks for some responses."""
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
    )


def _usage(msg: AIMessage) -> tuple[int, int]:
    um = getattr(msg, "usage_metadata", None) or {}
    return um.get("input_tokens", 0), um.get("output_tokens", 0)


def _clip(text: str, limit: int = _MAX_EVIDENCE_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _format_evidence(evidence: list[dict[str, Any]], with_tools: bool) -> str:
    if not evidence:
        return "(none yet)"
    blocks = []
    for i, e in enumerate(evidence, 1):
        block = f"[{i}] {e['agent']} — task: {e['task']}\nResult: {_clip(e['result'], 1500)}"
        if with_tools and e["tool_outputs"]:
            block += "\nRaw tool outputs:\n" + "\n".join(
                f"- {name}: {_clip(out)}" for name, out in e["tool_outputs"]
            )
        blocks.append(block)
    return "\n\n".join(blocks)


# -- graph ------------------------------------------------------------------

def _build_graph(trace: MultiAgentTrace, max_steps: int):
    """Compile the graph. Built per run so nodes can write to this run's trace."""
    specialists = {
        name: create_react_agent(_llm(2048), tools, prompt=_SYSTEM_BY_AGENT[name])
        for name, tools in _TOOLS_BY_AGENT.items()
    }

    def supervisor(state: MultiAgentState) -> dict[str, Any]:
        steps = state["steps_taken"] + 1

        # step cap: stop routing, answer with what we have
        if steps > max_steps:
            logger.warning(f"step cap ({max_steps}) hit, forcing finish")
            trace.step_cap_hit = True
            trace.add_step(SUPERVISOR, "route", "step cap hit", "finish", target=None)
            return {"next_agent": "finish", "task": "", "steps_taken": steps, "cap_hit": True}

        user = (
            f"Question: {state['question']}\n\n"
            f"Plan so far: {state['plan'] or '(none yet)'}\n\n"
            f"Evidence gathered:\n{_format_evidence(state['evidence'], with_tools=False)}\n\n"
            f"Steps used: {steps - 1} of {max_steps}\n"
        )
        if state["verifier_feedback"]:
            user += f"\nVerifier feedback on your last draft: {state['verifier_feedback']}\n"

        resp = _llm(512).invoke([SystemMessage(prompts.SUPERVISOR_SYSTEM), HumanMessage(user)])
        in_tok, out_tok = _usage(resp)
        decision = parse_json_object(_text(resp.content))

        if decision is None or decision.get("next") not in (*SPECIALISTS, "finish"):
            # unusable output: search first if we have nothing, otherwise answer
            logger.warning(f"supervisor returned unusable routing: {_text(resp.content)[:200]!r}")
            fallback = "finish" if state["evidence"] else "retrieval_agent"
            decision = {"next": fallback, "task": state["question"], "plan": [],
                        "reason": "unparseable supervisor output, used fallback"}

        target = decision["next"]
        task = (decision.get("task") or "").strip() or state["question"]
        plan = state["plan"] or [str(s) for s in decision.get("plan") or []]

        trace.add_step(
            SUPERVISOR, "route", user[-600:], decision.get("reason", ""),
            target=None if target == "finish" else target,
            llm_calls=1, input_tokens=in_tok, output_tokens=out_tok,
        )
        if target != "finish":
            trace.add_step(SUPERVISOR, "handoff", task, target=target)
        return {"next_agent": target, "task": task if target != "finish" else "",
                "plan": plan, "steps_taken": steps}

    def run_specialist(name: str):
        def node(state: MultiAgentState) -> dict[str, Any]:
            task = state["task"]
            if name == "analyst_agent":
                user = (f"Question: {state['question']}\n\nTask: {task}\n\n"
                        f"Evidence:\n{_format_evidence(state['evidence'], with_tools=True)}")
                resp = _llm(1024).invoke([SystemMessage(prompts.ANALYST_SYSTEM), HumanMessage(user)])
                in_tok, out_tok = _usage(resp)
                result, tool_outputs, llm_calls = _text(resp.content), [], 1
            else:
                out = specialists[name].invoke({"messages": [HumanMessage(task)]})
                result, tool_outputs, in_tok, out_tok, llm_calls = _record_react_run(name, out)

            trace.add_step(name, "draft", task, result, llm_calls=llm_calls,
                           input_tokens=in_tok, output_tokens=out_tok)
            trace.add_step(name, "handoff", result, target=SUPERVISOR)
            evidence = state["evidence"] + [{
                "agent": name, "task": task, "result": result, "tool_outputs": tool_outputs,
            }]
            return {"evidence": evidence, "steps_taken": state["steps_taken"] + 1}
        return node

    def _record_react_run(name: str, out: dict[str, Any]):
        """Pull tool calls, outputs, tokens and the final reply out of a ReAct run."""
        calls: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        in_tok = out_tok = llm_calls = 0
        result = ""
        for msg in out["messages"]:
            if isinstance(msg, AIMessage):
                llm_calls += 1
                i, o = _usage(msg)
                in_tok, out_tok = in_tok + i, out_tok + o
                for tc in msg.tool_calls or []:
                    calls[tc["id"]] = {"name": tc["name"], "args": tc["args"], "output": None}
                    order.append(tc["id"])
                if _text(msg.content).strip():
                    result = _text(msg.content)
            elif isinstance(msg, ToolMessage) and msg.tool_call_id in calls:
                calls[msg.tool_call_id]["output"] = msg.content
        tool_outputs = []
        for cid in order:
            c = calls[cid]
            trace.add_agent_tool_call(name, c["name"], c["args"], c["output"])
            tool_outputs.append((c["name"], str(c["output"])))
        return result, tool_outputs, in_tok, out_tok, llm_calls

    def draft(state: MultiAgentState) -> dict[str, Any]:
        user = (f"Question: {state['question']}\n\n"
                f"Evidence:\n{_format_evidence(state['evidence'], with_tools=True)}\n")
        if state["verifier_feedback"]:
            user += f"\nVerifier feedback on the previous draft: {state['verifier_feedback']}\n"
        resp = _llm(1024).invoke([SystemMessage(prompts.DRAFT_SYSTEM), HumanMessage(user)])
        in_tok, out_tok = _usage(resp)
        text = _text(resp.content).strip()
        trace.add_step(SUPERVISOR, "draft", state["question"], text, llm_calls=1,
                       input_tokens=in_tok, output_tokens=out_tok)
        return {"draft": text, "steps_taken": state["steps_taken"] + 1}

    def verifier(state: MultiAgentState) -> dict[str, Any]:
        user = (f"Question: {state['question']}\n\n"
                f"Evidence:\n{_format_evidence(state['evidence'], with_tools=True)}\n\n"
                f"Draft answer:\n{state['draft']}")
        resp = _llm(512).invoke([SystemMessage(prompts.VERIFIER_SYSTEM), HumanMessage(user)])
        in_tok, out_tok = _usage(resp)
        verdict = parse_json_object(_text(resp.content)) or {}
        action = "revise" if verdict.get("verdict") == "revise" else "approve"
        reason = str(verdict.get("reason", ""))
        trace.add_step(VERIFIER, "verify", state["draft"], f"{action}: {reason}".strip(": "),
                       llm_calls=1, input_tokens=in_tok, output_tokens=out_tok)
        if action == "revise":
            trace.add_step(VERIFIER, "handoff", reason, target=SUPERVISOR)
            return {"verifier_feedback": reason, "revise_count": state["revise_count"] + 1,
                    "steps_taken": state["steps_taken"] + 1}
        return {"verifier_feedback": "", "steps_taken": state["steps_taken"] + 1}

    def final(state: MultiAgentState) -> dict[str, Any]:
        trace.final_answer = state["draft"]
        trace.add_step(SUPERVISOR, "final", output=state["draft"])
        return {"answer": state["draft"]}

    # -- edges
    def after_supervisor(state: MultiAgentState) -> str:
        return "draft" if state["next_agent"] == "finish" else state["next_agent"]

    def after_draft(state: MultiAgentState) -> str:
        # verify once; after a revision, or at the step cap, go straight to final
        if state["cap_hit"] or state["revise_count"] >= MAX_REVISIONS:
            return "final"
        return "verifier_agent"

    def after_verifier(state: MultiAgentState) -> str:
        return "supervisor" if state["verifier_feedback"] else "final"

    g = StateGraph(MultiAgentState)
    g.add_node("supervisor", supervisor)
    for name in SPECIALISTS:
        g.add_node(name, run_specialist(name))
    g.add_node("draft", draft)
    g.add_node("verifier_agent", verifier)
    g.add_node("final", final)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", after_supervisor,
                            {**{n: n for n in SPECIALISTS}, "draft": "draft"})
    for name in SPECIALISTS:
        g.add_edge(name, "supervisor")
    g.add_conditional_edges("draft", after_draft, {"verifier_agent": "verifier_agent", "final": "final"})
    g.add_conditional_edges("verifier_agent", after_verifier, {"supervisor": "supervisor", "final": "final"})
    g.add_edge("final", END)
    return g.compile()


def run_multi_agent(question: str, max_steps: int = MAX_STEPS) -> MultiAgentResult:
    """Run a question through the supervisor + specialists. Mirrors run_agent()."""
    logger.info(f"multi-agent starting: {question!r}")
    t0 = time.perf_counter()
    trace = MultiAgentTrace(question=question)

    graph = _build_graph(trace, max_steps)
    state = graph.invoke(
        {"question": question, "plan": [], "evidence": [], "next_agent": "", "task": "",
         "draft": "", "verifier_feedback": "", "revise_count": 0, "steps_taken": 0,
         "cap_hit": False, "answer": ""},
        # generous: the step cap, not LangGraph's limit, is what bounds a run
        config={"recursion_limit": max_steps * 4 + 10},
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"multi-agent done in {latency_ms:.0f}ms — route={trace.route_sequence}, "
        f"{trace.llm_call_count} LLM calls, "
        f"{trace.total_input_tokens + trace.total_output_tokens} tokens"
    )
    return MultiAgentResult(question=question, answer=state["answer"], trace=trace,
                            latency_ms=latency_ms)
