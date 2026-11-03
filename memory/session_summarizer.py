"""
memory/session_summarizer.py

Day 16 — "memory write" side.

At the end of a session, summarize what was covered and persist it, so
Day 17's "if a paper was read yesterday, remember it" test passes.

Design choice: this module does NOT import an LLM client directly. It takes
an `llm_call` function as a parameter (Callable[[str], str]) so it's
provider-agnostic — plug in your own Anthropic/OpenAI call from wherever
your agent already makes LLM calls (planner.py, synthesizer.py). This also
makes it trivially testable with a stub, which is how it's tested below.

Wire-in point (end of your graph run / API request handler, roughly):

    from memory.session_summarizer import summarize_and_persist

    # after the graph finishes and you have the full turn history for this session:
    summarize_and_persist(
        store=store,
        turns=session_turns,       # list of {"query", "answer", "arxiv_ids"} dicts
        llm_call=my_anthropic_call_fn,
    )
    store.end_session(summary="handled in summarize_and_persist")
"""

import json
import re
from typing import Callable

from memory.store import MemoryStore

SUMMARY_PROMPT_TEMPLATE = """You are summarizing a research assistant session for long-term memory.

Below is the transcript of queries and answers from this session. Extract:
1. A one-paragraph summary of what was explored (for a human reading session history).
2. A list of distinct research topics covered (short phrases taken ONLY from what
   actually appears in the transcript — never invent topics, and never copy
   examples from these instructions).
3. A list of key conclusions/takeaways worth remembering for future sessions (concise, factual statements).

Respond ONLY with valid JSON, no other text, in this exact shape:
{{
  "summary": "...",
  "topics": ["...", "..."],
  "conclusions": [
    {{"text": "...", "topic": "...", "arxiv_id": "..."}}
  ]
}}
If an arxiv_id isn't clearly tied to a conclusion, omit that key or use null.

TRANSCRIPT:
{transcript}
"""


_TOPIC_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "in", "on", "to", "with",
    "from", "about", "into", "over", "as", "by", "at", "is", "are",
}


def _topic_supported_by_transcript(topic: str, transcript: str) -> bool:
    """
    Keep a topic only if most of its content words appear in the transcript.
    Stops the summarizer from copying instruction examples (or inventing
    adjacent jargon) into topics_covered.
    """
    words = [
        w for w in re.findall(r"[a-z0-9]+", (topic or "").lower())
        if w not in _TOPIC_STOPWORDS and len(w) >= 2
    ]
    if not words:
        return False
    haystack = (transcript or "").lower()
    hits = sum(1 for w in words if w in haystack)
    return hits >= max(1, (len(words) + 1) // 2)


def _filter_topics(topics: list, transcript: str) -> list:
    kept = []
    seen = set()
    for topic in topics or []:
        if not isinstance(topic, str):
            continue
        cleaned = topic.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if not _topic_supported_by_transcript(cleaned, transcript):
            continue
        seen.add(key)
        kept.append(cleaned)
    return kept


def _build_transcript(turns: list[dict]) -> str:
    lines = []
    for i, turn in enumerate(turns, 1):
        lines.append(f"[Turn {i}]")
        lines.append(f"Query: {turn.get('query', '')}")
        lines.append(f"Answer: {turn.get('answer', '')}")
        if turn.get("arxiv_ids"):
            lines.append(f"Papers referenced: {', '.join(turn['arxiv_ids'])}")
        lines.append("")
    return "\n".join(lines)


def summarize_session(turns: list[dict], llm_call: Callable[[str], str]) -> dict:
    """
    Pure function: turns -> structured summary dict. Doesn't touch storage,
    so it's easy to unit test independent of SQLite.
    """
    if not turns:
        return {"summary": "Empty session — nothing to summarize.", "topics": [], "conclusions": []}

    transcript = _build_transcript(turns)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
    raw = llm_call(prompt)

    # Be defensive: strip markdown code fences if the model adds them despite instructions.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: don't crash the session teardown just because the summarizer
        # LLM call misbehaved. Persist raw text as the summary, no structured extraction.
        return {"summary": raw.strip(), "topics": [], "conclusions": []}

    parsed.setdefault("summary", "")
    parsed.setdefault("topics", [])
    parsed.setdefault("conclusions", [])
    parsed["topics"] = _filter_topics(parsed.get("topics"), transcript)
    # Also drop conclusions whose topic is unsupported / empty text.
    safe_conclusions = []
    for c in parsed.get("conclusions") or []:
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        if not _topic_supported_by_transcript(text, transcript):
            continue
        safe_conclusions.append(c)
    parsed["conclusions"] = safe_conclusions
    return parsed


def summarize_and_persist(
    store: MemoryStore,
    turns: list[dict],
    llm_call: Callable[[str], str],
) -> dict:
    """
    Full write path: summarize, then persist topics/conclusions/session summary
    to the MemoryStore. Call this once per session, right before/at teardown.
    Returns the structured summary dict in case the caller wants to log it.
    """
    result = summarize_session(turns, llm_call)

    for topic in result.get("topics", []):
        store.record_topic(topic)

    for c in result.get("conclusions", []):
        store.record_conclusion(
            conclusion_text=c.get("text", ""),
            topic=c.get("topic"),
            arxiv_id=c.get("arxiv_id"),
        )

    store.end_session(summary=result.get("summary", ""))
    return result