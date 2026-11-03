"""
memory/context_injection.py

Day 16 — "memory read" side.

Takes the MemorySnapshot produced by MemoryStore.get_memory_snapshot() and
turns it into plain text you inject into the planner's prompt/context at the
start of a session, so the planner can decide things like:
    - "user already read this paper -> don't re-fetch, use cached knowledge"
    - "user already covered this topic -> maybe just synthesize from memory"
    - "user prefers concise answers -> keep synthesis short"

Wire-in point (your planner.py, roughly):

    from memory.store import MemoryStore
    from memory.context_injection import format_memory_context

    store = MemoryStore()
    store.start_session(session_id)
    snapshot = store.get_memory_snapshot()
    memory_context = format_memory_context(snapshot)

    # then prepend/inject into whatever you pass as system/context to the planner LLM call:
    planner_system_prompt = BASE_PLANNER_PROMPT + "\\n\\n" + memory_context

If your AgentState is a TypedDict/pydantic model, add a `memory_context: str`
field and set it once at graph entry, so every node (planner, synthesizer)
can read it without re-querying the DB.
"""

from memory.schema import MemorySnapshot


def format_memory_context(snapshot: MemorySnapshot) -> str:
    """
    Returns a compact, LLM-readable block. Kept deliberately short —
    this goes into every planner call, so don't dump full paper abstracts
    or long conclusion text here; that's what retrieval/RAG is for.
    """
    if not any([snapshot.recent_topics, snapshot.recent_conclusions,
                snapshot.known_paper_ids, snapshot.preferences]):
        return "MEMORY: No prior session history. This is a fresh start."

    lines = ["MEMORY FROM PAST SESSIONS:"]

    if snapshot.recent_topics:
        lines.append("- Topics previously explored:")
        for c in snapshot.recent_topics:
            lines.append(f"  • {c}")

    if snapshot.known_paper_ids:
        ids = sorted(snapshot.known_paper_ids)
        ids_preview = ", ".join(ids[:10])
        suffix = " (and more)" if len(snapshot.known_paper_ids) > 10 else ""
        lines.append(f"- Papers already read (arxiv IDs): {ids_preview}{suffix}")
        # lines.append(
        #     "  -> If the user asks about one of these, use cached knowledge "
        #     "instead of re-fetching, unless they explicitly ask to re-check it."
        # )

    if snapshot.recent_conclusions:
        lines.append("- Key conclusions from past sessions:")
        for c in snapshot.recent_conclusions:
            lines.append(f"  • {c}")

    if snapshot.preferences:
        lines.append("- User Preferences:")
        for key, value in snapshot.preferences.items():
            pretty_key = key.replace("_", " ").title()
            lines.append(f"  • {pretty_key}: {value}")

    return "\n".join(lines)


def known_paper_notice(arxiv_id: str, snapshot: MemorySnapshot) -> str | None:
    """
    Convenience check for the planner/tool-router: call this before deciding
    to hit paper_fetch.py for a given arxiv_id. Returns a short notice string
    if already known, or None if it should be fetched fresh.

    Usage in your planner's routing logic:

        notice = known_paper_notice(arxiv_id, snapshot)
        if notice:
            # skip search_arxiv / paper_fetch node, go straight to synthesizer
            ...
    """
    if arxiv_id in snapshot.known_paper_ids:
        return f"Paper {arxiv_id} was already read in a previous session — using cached data."
    return None