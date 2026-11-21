"""
MCP server exposing three existing agent tools over stdio.

The tools are not re-implemented: each MCP tool forwards to the LangChain tool in
src/agent/tools.py, so there is one copy of the query logic. The input schema is read
from the original function's signature (functools.wraps), so it cannot drift either.

    python -m src.mcp_server.server        # speaks MCP on stdin/stdout

stdout carries the protocol, so nothing here may print to it. loguru writes to stderr.
"""

import functools
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from src.agent import tools as T

mcp = FastMCP("probeai-movies")

# the subset exposed over MCP
EXPOSED = (T.search_by_director, T.filter_movies, T.get_movie_details)


def _expose(lc_tool: Any) -> Callable[..., Any]:
    """Register `lc_tool` (a LangChain StructuredTool) as an MCP tool that calls it."""

    @functools.wraps(lc_tool.func)  # keeps the name, docstring and type hints -> MCP schema
    def forward(**kwargs: Any) -> Any:
        # invoke() validates arguments and calls tool.func, exactly like the in-process agent
        return lc_tool.invoke(kwargs)

    mcp.tool(name=lc_tool.name, description=lc_tool.description)(forward)
    return forward


for _tool in EXPOSED:
    _expose(_tool)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
