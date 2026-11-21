"""
Load the agent's tools from an MCP server instead of importing them.

One persistent session is used for the whole run: MultiServerMCPClient.get_tools() would start a
new server process for every tool call (each one re-imports the project, several seconds).
The returned tools are async-only, so the agent that uses them must be run with astream/ainvoke.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

ROOT = Path(__file__).resolve().parents[2]
SERVER_NAME = "probeai"


def default_server() -> dict[str, Any]:
    """Spawn our own server over stdio. cwd is the repo root so it finds .env and the `src` package."""
    return {"transport": "stdio", "command": sys.executable, "args": ["-m", "src.mcp_server.server"],
            "cwd": str(ROOT)}


@asynccontextmanager
async def mcp_tools(server: dict[str, Any] | None = None) -> AsyncIterator[list[BaseTool]]:
    """Yield the MCP server's tools as LangChain tools; the server process lives for the `with` block."""
    client = MultiServerMCPClient({SERVER_NAME: server or default_server()})
    async with client.session(SERVER_NAME) as session:
        yield await load_mcp_tools(session)
