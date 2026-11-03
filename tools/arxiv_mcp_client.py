"""
tools/arxiv_mcp_client.py

Day 19 -- the client-side counterpart to mcp_server/arxiv_mcp_server.py's
search_arxiv_tool. This is what agents/tool_nodes.py calls now, instead of
importing tools.arxiv_search.search_arxiv directly.

Design note: opens a fresh stdio subprocess + session per call. That's
wasteful (subprocess startup cost on every search) but simplest to get
correct first. Day 20's plan explicitly budgets time for protocol/connection
fixes -- a persistent, reused client session across the graph's lifetime is
the natural follow-up optimization, not something to solve today.
"""

import asyncio
import os
import sys
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "mcp_server", "arxiv_mcp_server.py"
)


async def _search_arxiv_async(
    topic: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[Dict]:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(
                session.call_tool("search_arxiv_tool", {
                    "topic": topic,
                    "max_results": max_results,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                }),
                timeout=20,  # fail fast instead of hanging the whole eval run
            )

    if result.isError:
        # Surface the server's error text rather than a bare truthy/falsy failure --
        # matches how search_arxiv_node's except block expects to log a message.
        error_text = result.content[0].text if result.content else "unknown MCP tool error"
        raise RuntimeError(f"search_arxiv_tool MCP call failed: {error_text}")

    # Prefer structuredContent (typed, no re-parsing needed) over the text
    # serialization FastMCP also includes for backward compatibility.
    if result.structuredContent and "result" in result.structuredContent:
        return result.structuredContent["result"]

    # Fallback: shouldn't normally hit this given FastMCP's List[Dict] return
    # type, but don't silently return nothing if the SDK's wrapping changes.
    import json
    return json.loads(result.content[0].text)


def search_arxiv_via_mcp(
    topic: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[Dict]:
    """
    Sync wrapper -- agents/tool_nodes.py's search_arxiv_node is a sync
    function (LangGraph node), so this bridges into the async MCP client
    with asyncio.run(). Returns the same list-of-dicts shape
    tools.arxiv_search.search_arxiv's ArxivPaper.to_dict() produces.
    """
    return asyncio.run(
        _search_arxiv_async(topic, max_results, sort_by, sort_order)
    )


if __name__ == "__main__":
    papers = search_arxiv_via_mcp(
        topic="LoRA fine-tuning",
        max_results=3,
    )

    for i, paper in enumerate(papers, 1):
        print(f"\nPaper {i}")
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"URL: {paper['url']}")
        print("-" * 60)