"""
Comparator node: given 2+ papers already in retrieved_docs, produce a
structured comparison (method, results, limitations) instead of a
single narrative answer.
"""

import requests
from collections import OrderedDict
from typing import List, Dict

from agents.synthesizer import (
    fix_literal_n_citations,
    clamp_citation_indices,
    renumber_sources_to_citations,
    ollama_chat_with_deadline,
    _source_referenced,
)
from rag.retrieve import split_comparison_aspects, aspect_keywords, _keywords

OLLAMA_URL = "http://localhost:11434/api/chat"
COMPARATOR_MODEL = "llama3.1:8b"
# Keep comparator prompts bounded — 3 full chunks on CPU llama3.1:8b can
# spend 15+ minutes on prefill alone (seen on q09 at ~1100s).
MAX_CHUNK_CHARS = 900
MAX_PAPERS_TO_COMPARE = 2  # named A-vs-B compares rarely need a 3rd distractor
OLLAMA_NUM_PREDICT = 512

COMPARATOR_SYSTEM_PROMPT = """You are a research assistant answering a user's question by comparing multiple papers.
You will be given excerpts from 2 or more papers.

Rules:
- Write a cohesive, flowing comparative response that directly answers the user's specific question.
- Do NOT just list independent summaries of each paper. Synthesize the findings and compare them dynamically.
- Only use information present in the excerpts.
- Cite sources inline using the excerpt's ACTUAL numbers, e.g. [1] or [2]. Never write the literal text [n].
- Write entirely in your own words — do not copy raw equations or notation from the excerpts.
- Output ONLY the final plain-text response. Do not use JSON, XML, or special formatting headers.
- Keep the comparison focused — do not drag in unrelated papers beyond the excerpts provided.
- Be concise: aim for 1-3 short paragraphs, not a long survey.
"""


def group_chunks_by_paper(chunks: List[Dict]) -> "OrderedDict[str, List[Dict]]":
    grouped = OrderedDict()
    for chunk in chunks:
        grouped.setdefault(chunk["arxiv_id"], []).append(chunk)
    return grouped


def _truncate(text: str, limit: int = MAX_CHUNK_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " ..."


def build_context_block(grouped: "OrderedDict[str, List[Dict]]", max_chunks_per_paper: int = 1) -> str:
    lines = []
    for i, (arxiv_id, paper_chunks) in enumerate(grouped.items(), 1):
        title = paper_chunks[0]["title"]
        # Abstract first — it summarizes method + results, the most useful
        # single chunk for a comparison (stable sort keeps retrieval order).
        ordered = sorted(
            paper_chunks, key=lambda c: 0 if c.get("source") == "abstract" else 1
        )
        limited_chunks = ordered[:max_chunks_per_paper]
        excerpt_text = "\n---\n".join(_truncate(c["text"]) for c in limited_chunks)
        lines.append(f"[{i}] Source: {title} ({arxiv_id})\n{excerpt_text}\n")
    return "\n".join(lines)


def _paper_matches_aspect(title: str, aspect: str) -> bool:
    # Use distinctive tokens only — "fine"/"tuning" match every title in a
    # fine-tuning corpus and would make any paper "match" any aspect.
    title_kw = set(_keywords(title))
    aspect_kw = set(aspect_keywords(aspect))
    if not aspect_kw:
        return False
    return len(title_kw & aspect_kw) >= max(1, (len(aspect_kw) + 1) // 2)


def _select_papers_for_comparison(
    grouped: "OrderedDict[str, List[Dict]]", query: str
) -> "OrderedDict[str, List[Dict]]":
    """
    Prefer one paper per named comparison aspect when the query names two
    sides (e.g. "A vs B"). Falls back to retrieval order otherwise.
    """
    aspects = split_comparison_aspects(query)
    items = list(grouped.items())
    if len(aspects) < 2 or len(items) <= 2:
        return OrderedDict(items[:MAX_PAPERS_TO_COMPARE])

    selected: "OrderedDict[str, List[Dict]]" = OrderedDict()
    remaining = list(items)

    for aspect in aspects:
        match_idx = next(
            (
                i
                for i, (_, chunks) in enumerate(remaining)
                if _paper_matches_aspect(chunks[0].get("title", ""), aspect)
            ),
            None,
        )
        if match_idx is None:
            continue
        arxiv_id, chunks = remaining.pop(match_idx)
        selected[arxiv_id] = chunks

    for arxiv_id, chunks in remaining:
        if len(selected) >= MAX_PAPERS_TO_COMPARE:
            break
        selected[arxiv_id] = chunks

    # Need at least 2 papers to compare; if aspect matching collapsed too far,
    # fall back to original order.
    if len(selected) < 2:
        return OrderedDict(items[:MAX_PAPERS_TO_COMPARE])
    return selected


def compare_papers_node(state: dict) -> dict:
    query = state["query"]
    chunks = state.get("retrieved_docs", [])
    grouped = group_chunks_by_paper(chunks)

    if len(grouped) < 2:
        return {
            "answer": "I need at least 2 distinct papers already retrieved to compare — "
                      "try asking me to search or retrieve relevant papers first.",
            "sources": [],
            "tool_calls": state.get("tool_calls", []) + ["comparator(insufficient_papers)"],
        }

    grouped = _select_papers_for_comparison(grouped, query)

    context_block = build_context_block(grouped)
    user_prompt = f"""Question: {query}

Excerpts:
{context_block}

Compare these papers as instructed. If the question asks which is "better",
explain that the approaches may solve different problems and report tradeoffs
rather than declaring a universal winner. Only cite excerpt numbers that appear
above (1..{len(grouped)})."""

    try:
        raw_answer = ollama_chat_with_deadline(
            COMPARATOR_MODEL,
            [
                {"role": "system", "content": COMPARATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            num_predict=OLLAMA_NUM_PREDICT,
            node_name="comparator",
        )
        n_sources = len(grouped)
        answer_text = clamp_citation_indices(
            fix_literal_n_citations(raw_answer.strip(), n_sources),
            n_sources,
        )

    except (requests.ConnectionError, requests.Timeout, requests.HTTPError, KeyError) as e:
        print(f"[compare_papers_node] error: {e}")
        answer_text = f"[ERROR: comparison failed — {e}]"

    all_sources = [
        {"n": i, "title": paper_chunks[0].get("title", ""), "url": paper_chunks[0].get("url", ""), "arxiv_id": arxiv_id}
        for i, (arxiv_id, paper_chunks) in enumerate(grouped.items(), 1)
    ]

    sources = [s for s in all_sources if _source_referenced(s, answer_text)]
    if not sources and all_sources and answer_text and not answer_text.startswith("[ERROR"):
        sources = all_sources[:2]

    answer_text, sources = renumber_sources_to_citations(answer_text, sources)

    return {
        "answer": answer_text,
        "sources": sources,
        "tool_calls": state.get("tool_calls", []) + [f"comparator({len(grouped)} papers)"],
    }