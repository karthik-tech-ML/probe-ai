"""
Smoke test for the MCP server: start it over stdio, list its tools, call each once.

    python scripts/smoke_test_mcp.py

Needs the database running (the tools query it). No LLM calls, so it costs nothing.
Exits non-zero if a tool is missing, errors, or returns nothing.
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"search_by_director", "filter_movies", "get_movie_details"}

# one call per tool; Inception's TMDB id is 27205
CALLS = [
    ("search_by_director", {"director_name": "Christopher Nolan"}),
    ("filter_movies", {"genre": "Comedy", "min_rating": 7.5, "year_from": 2010, "limit": 5}),
    ("get_movie_details", {"tmdb_id": 27205}),
]


def server_params() -> StdioServerParameters:
    # cwd is the repo root so the server finds .env and the `src` package
    return StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server.server"], cwd=str(ROOT))


def summarize(result) -> str:
    """One line describing a tool result: structured items if present, else the text length."""
    data = result.structuredContent
    if isinstance(data, dict) and "result" in data:
        data = data["result"]
    if isinstance(data, list):
        first = data[0].get("title") if data and isinstance(data[0], dict) else None
        return f"{len(data)} items, first: {first!r}"
    if isinstance(data, dict):
        return f"object with keys {sorted(data)[:4]}, title: {data.get('title')!r}"
    text = "".join(getattr(c, "text", "") for c in result.content)
    return f"text, {len(text)} chars"


async def main() -> int:
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            print(f"tools: {sorted(names)}")
            if names != EXPECTED:
                print(f"FAIL: expected {sorted(EXPECTED)}")
                return 1
            failures = 0
            for name, args in CALLS:
                result = await session.call_tool(name, args)
                ok = not result.isError and bool(result.content)
                failures += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {name}({json.dumps(args)}) -> "
                      f"{summarize(result) if ok else result.content}")
            return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
