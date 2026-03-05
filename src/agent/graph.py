"""
LangGraph ReAct agent for multi-step movie questions.

Uses Claude as the reasoning model with structured tools for
querying the movie database. Solves the retrieval problem for
multi_field and aggregation queries where single-shot vector
search falls short.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from loguru import logger

from src.agent.memory import AgentTrace
from src.agent.tools import ALL_TOOLS
from src.config import settings

_SYSTEM_PROMPT = """\
You are a movie knowledge assistant with access to a database of ~4800 movies \
(the TMDB 5000 dataset). You have tools to search and filter the database.

Rules:
1. ONLY answer based on data returned by your tools. Never use outside knowledge.
2. If your tools return no results, say you don't have that information.
3. Pick the most specific tool for the job:
   - search_by_director, search_by_cast, search_by_genre for single-field lookups
   - filter_movies when combining multiple constraints (genre + year + rating, etc.)
   - search_movies for vague or topic-based queries
   - get_movie_details when you need full info on a specific movie
4. For cross-referencing questions (e.g. "actors in both X and Y directors' films"), \
call the relevant tools for each side, then compare the results yourself.
5. Keep answers concise and grounded in the tool results.
"""


@dataclass
class AgentResult:
    question: str
    answer: str
    trace: AgentTrace
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "latency_ms": round(self.latency_ms, 1),
            "trace": self.trace.to_dict(),
        }


def _build_agent():
    """Create the LangGraph ReAct agent with our tools and Claude."""
    model = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        api_key=settings.anthropic_api_key,
        max_tokens=2048,
    )
    return create_react_agent(model, ALL_TOOLS, prompt=_SYSTEM_PROMPT)


def run_agent(question: str) -> AgentResult:
    """
    Run a question through the ReAct agent.

    The agent will:
    1. Decide which tools to call
    2. Call them (possibly multiple rounds)
    3. Synthesize a final answer from tool results

    Returns the answer plus a full trace of tool calls for eval.
    """
    logger.info(f"agent starting: {question!r}")
    t0 = time.perf_counter()

    agent = _build_agent()
    trace = AgentTrace(question=question)

    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    # walk through the message history to build the trace
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            # accumulate token usage
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                trace.total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
                trace.total_output_tokens += msg.usage_metadata.get("output_tokens", 0)

            # record tool calls the model decided to make
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    # the actual output comes in the next ToolMessage,
                    # we'll patch it in below
                    trace.add_tool_call(
                        name=tc["name"],
                        input=tc["args"],
                        output=None,  # filled in from ToolMessage
                    )

            # the last AIMessage with text content is the final answer
            if isinstance(msg.content, str) and msg.content.strip():
                trace.final_answer = msg.content

        elif isinstance(msg, ToolMessage):
            # patch the output into the most recent tool call with matching name
            # that still has output=None
            for tc in reversed(trace.tool_calls):
                if tc.tool_output is None:
                    tc.tool_output = msg.content
                    break

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"agent done in {latency_ms:.0f}ms — "
        f"{trace.num_tool_calls} tool calls, "
        f"{trace.total_input_tokens + trace.total_output_tokens} total tokens"
    )

    return AgentResult(
        question=question,
        answer=trace.final_answer,
        trace=trace,
        latency_ms=latency_ms,
    )
