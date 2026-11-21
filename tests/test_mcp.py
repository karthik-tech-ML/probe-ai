"""MCP prototype: server tool surface, forwarding to the real tools, and the agent flag. No LLM calls."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.mcp_server.server import EXPOSED, mcp


def test_server_exposes_exactly_the_three_tools_with_original_schemas():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert set(tools) == {"search_by_director", "filter_movies", "get_movie_details"}
    assert tools["search_by_director"].inputSchema["required"] == ["director_name"]
    # schema comes from the original signature: every filter_movies parameter is present
    from src.agent.tools import filter_movies
    assert set(tools["filter_movies"].inputSchema["properties"]) == set(filter_movies.args)
    assert [t.name for t in EXPOSED] == ["search_by_director", "filter_movies", "get_movie_details"]


def test_mcp_tool_forwards_to_the_existing_tool_without_duplicating_logic():
    from src.agent import tools as T
    seen = []
    original = T.search_by_director.func
    T.search_by_director.func = lambda director_name: seen.append(director_name) or [{"title": "X"}]
    try:
        asyncio.run(mcp.call_tool("search_by_director", {"director_name": "Nolan"}))
    finally:
        T.search_by_director.func = original
    assert seen == ["Nolan"]  # the call went through the in-process tool's func


def test_default_server_spec_is_stdio_from_the_repo_root():
    from src.agent.mcp_tools import ROOT, default_server
    spec = default_server()
    assert spec["transport"] == "stdio" and spec["args"] == ["-m", "src.mcp_server.server"]
    assert spec["cwd"] == str(ROOT) and (ROOT / "src").is_dir()


def test_mcp_tools_load_from_a_real_server_process():
    from src.agent.mcp_tools import mcp_tools

    async def names():
        async with mcp_tools() as tools:
            return sorted(t.name for t in tools)

    assert asyncio.run(names()) == ["filter_movies", "get_movie_details", "search_by_director"]


class _FakeAgent:
    """Stands in for the ReAct agent: one tool call, then a final answer."""

    def stream(self, inputs, config=None, stream_mode=None):
        yield {"messages": [
            HumanMessage("q"),
            AIMessage("", tool_calls=[{"name": "search_by_director", "args": {"director_name": "x"}, "id": "1"}]),
            ToolMessage("[]", tool_call_id="1"),
            AIMessage("final answer"),
        ]}


def test_run_agent_default_path_is_unchanged_and_never_touches_mcp(monkeypatch):
    import src.agent.graph as G

    monkeypatch.setattr(G, "_build_agent", lambda tools=None: _FakeAgent())
    monkeypatch.setattr(G, "_run_with_mcp", lambda *a, **k: (_ for _ in ()).throw(AssertionError("MCP used")))
    res = G.run_agent("q")
    assert res.answer == "final answer"
    assert res.trace.tool_calls[0].tool_name == "search_by_director" and res.trace.tool_calls[0].tool_output == "[]"


def test_run_agent_flag_routes_to_the_mcp_path(monkeypatch):
    import src.agent.graph as G

    async def fake(question, server):
        assert server == {"marker": 1}
        return {"messages": [HumanMessage("q"), AIMessage("via mcp")]}, False

    monkeypatch.setattr(G, "_run_with_mcp", fake)
    monkeypatch.setattr(G, "_build_agent", lambda tools=None: (_ for _ in ()).throw(AssertionError("default used")))
    assert G.run_agent("q", use_mcp=True, mcp_server={"marker": 1}).answer == "via mcp"
