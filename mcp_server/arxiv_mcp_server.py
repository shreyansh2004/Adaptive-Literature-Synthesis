"""
mcp_server/arxiv_mcp_server.py

Day 18 — minimal MCP server skeleton.

This is deliberately NOT wired to your real tools/arxiv_search.py yet.
Day 18's goal is just: learn the server/client model, get a server process
that speaks MCP correctly and exposes at least one discoverable tool.
Day 19 replaces `ping` below with your real arxiv_search logic.

Transport: stdio. Your MCP client launches this file as a subprocess and
talks to it over stdin/stdout using JSON-RPC framing — the SDK handles all
the framing/protocol details; you only write plain Python functions.

Run standalone (for a sanity check, not the normal usage pattern):
    python3 mcp_server/arxiv_mcp_server.py
    (it'll just sit there waiting on stdio — Ctrl+C to exit. Real usage is
    always via an MCP client, like tests/test_mcp_connection.py alongside this file.)
"""

import sys
import os
from typing import List, Dict

# Make tools/ importable when this file is launched directly as a subprocess
# (its cwd/sys.path won't automatically include your repo root otherwise).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP
from tools.arxiv_search import search_arxiv

# The server's name is what shows up when a client lists connected servers.
mcp = FastMCP("arxiv-research-tools")


@mcp.tool()
def ping(message: str) -> str:
    """
    Placeholder tool to prove the server/client wiring works end to end.
    Kept around as a cheap liveness check independent of arXiv/network.
    """
    return f"pong: {message}"


@mcp.tool()
def search_arxiv_tool(
    topic: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[Dict]:
    """
    Search arXiv for papers on a topic. Returns structured paper metadata:
    arxiv_id, title, abstract, authors, url, pdf_url, published, updated.

    This wraps tools.arxiv_search.search_arxiv exactly as-is -- no behavior
    change, just a new call boundary (MCP tool call instead of direct
    Python import). Same retries/rate-limit handling as before.
    """
    papers = search_arxiv(
        topic=topic,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return [p.to_dict() for p in papers]


if __name__ == "__main__":
    # stdio transport: this process reads/writes JSON-RPC over stdin/stdout.
    # A client launches this exact command as a subprocess to talk to it.
    mcp.run(transport="stdio")