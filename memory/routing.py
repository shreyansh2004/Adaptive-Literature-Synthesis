"""
memory/routing.py

Day 17 — memory-aware behavior in the planner's routing decision.

This is the piece that actually changes agent *behavior*: given a query and
the current MemorySnapshot, decide whether to hit search_arxiv/paper_fetch,
or skip straight to synthesis using cached knowledge.

This is intentionally a thin, testable layer that sits in front of your
existing planner routing (Day 9's planner node). Call `decide_next_action`
first; if it returns "use_cached_paper", short-circuit your normal
search_arxiv -> paper_fetch tool chain and go straight to the synthesizer
with the cached PaperRecord. Otherwise, fall through to your existing
LLM-based planner decision as before.

"""

import re
from typing import Optional

from memory.schema import MemorySnapshot

# Matches standard arXiv ID formats: 2106.09685, 2106.09685v2, arXiv:2106.09685
ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")

# "Have you already read the LoRA paper?" / "did you read paper 2106.09685?"
READ_PAPER_PATTERNS = [
    re.compile(
        r"(?:have you (?:already )?read|already read|did you read|have you seen)"
        r".{0,60}?(?:the\s+)?([a-zA-Z][\w\-]*(?:\s+[a-zA-Z][\w\-]*){0,3})\s+paper",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:have you (?:already )?read|already read|did you read|have you seen)"
        r".{0,40}?paper\s+(?:arxiv:)?(\d{4}\.\d{4,5})",
        re.IGNORECASE,
    ),
]


def extract_arxiv_id(query: str) -> Optional[str]:
    """
    Best-effort extraction of an explicit arxiv ID from a user query.
    Returns None if the query doesn't mention one directly (e.g. topic-only
    queries like "find papers on LoRA" — those still go through normal search).
    """
    match = ARXIV_ID_PATTERN.search(query)
    return match.group(1) if match else None


MEMORY_RECALL_KEYWORDS = {
    "what did we discuss", "what did we just discuss", "what have we covered",
    "what did we talk about", "earlier you said", "you mentioned",
    "recap", "summarize our", "summarize what we", "summarize what i", "summarize what you",
}


def query_requests_memory_recall(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in MEMORY_RECALL_KEYWORDS)


def query_asks_if_paper_read(query: str) -> bool:
    q = query.lower()
    return any(
        phrase in q
        for phrase in (
            "have you already read",
            "have you read",
            "already read the",
            "did you read",
            "have you seen the",
        )
    )


def extract_paper_read_lookup(query: str) -> Optional[str]:
    """
    Extract a title keyword or arxiv id from a 'have you read X paper?' query.
    Returns None if the query isn't that shape.
    """
    for pattern in READ_PAPER_PATTERNS:
        match = pattern.search(query)
        if match:
            return match.group(1).strip()
    return None


def decide_next_action(query: str, snapshot: MemorySnapshot) -> dict:
    """
    Returns a dict:
        {
            "action": "use_cached_paper" | "report_paper_read" | "fall_through",
            "arxiv_id": str | None,
            "title_keyword": str | None,
            "notice": str | None,   # human-readable, safe to show/prepend to the final answer
        }

    "fall_through" means: no memory shortcut applies, let the normal
    Day 9 planner (search_arxiv / retrieve_local / answer_now) decide.
    """
    arxiv_id = extract_arxiv_id(query)

    if arxiv_id and arxiv_id in snapshot.known_paper_ids:
        return {
            "action": "use_cached_paper",
            "arxiv_id": arxiv_id,
            "title_keyword": None,
            "notice": (
                f"I've already read paper {arxiv_id} in a previous session — "
                f"using what I remember instead of re-fetching it."
            ),
        }

    if query_asks_if_paper_read(query):
        lookup = extract_paper_read_lookup(query) or arxiv_id
        return {
            "action": "report_paper_read",
            "arxiv_id": arxiv_id,
            "title_keyword": lookup,
            "notice": None,  # planner fills this after looking up the store
        }

    return {
        "action": "fall_through",
        "arxiv_id": arxiv_id,
        "title_keyword": None,
        "notice": None,
    }


def topic_already_covered_notice(topic: str, snapshot: MemorySnapshot) -> Optional[str]:
    """
    Softer, topic-level version — doesn't skip a tool call (topics are broader
    than a single paper and probably still worth a fresh search), but gives the
    planner/synthesizer a notice it can surface to the user, e.g.
    "You explored this topic before; here's what's new since then."
    Purely informational — caller decides whether/how to use it.
    """
    normalized = topic.strip().lower()
    for past_topic in snapshot.recent_topics:
        if past_topic.strip().lower() == normalized:
            return f"You explored '{past_topic}' in a previous session."
    return None
