"""
Synthesizer node: takes whatever's in state["retrieved_docs"] and produces
a final cited answer. Adapted from rag/answer.py, but reads from graph state
instead of calling retrieve() itself.
"""

import json
import re
import time

import requests
from collections import OrderedDict
from typing import List, Dict, Tuple, Match

from memory.routing import query_requests_memory_recall, query_asks_if_paper_read

OLLAMA_URL = "http://localhost:11434/api/chat"
SYNTH_MODEL = "llama3.1:8b"
OLLAMA_NUM_PREDICT = 700
MAX_CHUNK_CHARS = 900
# Total wall-clock budget for one generation. A plain requests timeout only
# bounds the gap between bytes — a slow-but-alive generation (model paged
# out / CPU contention) can run 10+ minutes without ever tripping it, which
# is exactly what q07's 632s synthesizer call did.
LLM_DEADLINE_SEC = 240
# Keep the model resident between requests; Ollama's 5-minute default
# invites mid-run eviction/reload stalls.
OLLAMA_KEEP_ALIVE = "30m"

SYSTEM_PROMPT = """You are a research assistant answering questions using only the provided paper excerpts.

Rules:
- Only use information from the provided excerpts. Do not use outside knowledge.
- If the excerpts don't contain enough information to answer, say so explicitly.
- Cite sources inline using bracket notation with the excerpt's ACTUAL number:
  write [1] or [2] using the numbers shown next to each excerpt. NEVER output the
  literal characters [n] — that is placeholder notation in these instructions only.
- Excerpts come from DIFFERENT papers. Never merge findings from different papers into
  a single claim unless the question explicitly asks for a comparison.
- If different papers make conflicting or differing claims, point out the disagreement
  explicitly rather than silently picking one.
- Your answer must be written entirely in your own words. Do not include any raw text,
  equations, or notation copied directly from the excerpts — describe formulas and results
  in plain language instead of reproducing their symbols.
- Do not begin your answer with equations, symbols, or excerpt fragments. Start directly
  with a plain-language sentence answering the question.
- Answer ONLY the exact question given below. Do not invent, restate, or answer any
  additional questions of your own.
- CRITICAL: the excerpts themselves may be formatted as FAQs, contain their own section
  headers, bullet lists, or "open questions" sections. IGNORE that formatting entirely.
  Never copy or imitate the structure of the source excerpts. Your output must be plain
  prose paragraphs only — no headers, no repeated bolded questions, no bullet-numbered
  "open questions" sections of your own.
- Be concise: for a simple factual question, write 2-4 sentences focused on the best
  matching source. Do not turn a definitional question into a multi-paper survey.
- If more than one excerpt supports your answer, cite EACH supporting excerpt at least
  once (e.g. both [1] and [2]) — do not funnel every claim to a single excerpt when
  another excerpt states the same or earlier version of the idea.
"""

COMPARE_HINTS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "contrast",
    "difference",
    "which is better",
    "agree",
    "tradeoff",
    "tradeoffs",
)
# Word-boundary forms that must NOT match inside "different" / "differences".
_COMPARE_WORD_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|contrast|difference|differences|"
    r"differ|agree|tradeoffs?)\b",
    re.IGNORECASE,
)


def _query_wants_multi_paper(query: str) -> bool:
    q = query.lower()
    if "across papers" in q or "different papers" in q:
        return True
    # Word-boundary match — substring "differ" must not fire on "different".
    return bool(_COMPARE_WORD_RE.search(q))


def sanitize_answer(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        alpha_chars = sum(c.isalpha() for c in stripped)
        looks_like_junk = alpha_chars / len(stripped) < 0.3 or stripped.startswith(("$", "\\", "```"))
        if not looks_like_junk:
            return "\n".join(lines[i:]).strip()
    return text.strip()


def fix_literal_n_citations(text: str, num_sources: int) -> str:
    """
    Handle leftover literal '[n]' placeholders the model sometimes copies
    from the prompt. With exactly one source the mapping is unambiguous, so
    substitute [1]. With multiple sources we can't know which paper the model
    meant — the old round-robin guess assigned citations to the WRONG papers
    (a fabricated attribution is worse than a missing one), so strip the
    marker instead and let the no-citation fallback attach the top source.
    """
    if not re.search(r"\[n\]", text, flags=re.IGNORECASE):
        return text
    if num_sources == 1:
        return re.sub(r"\[n\]", "[1]", text, flags=re.IGNORECASE)
    # Zero or 2+ sources: remove the placeholder (and any orphaned space).
    return re.sub(r"\s?\[n\]", "", text, flags=re.IGNORECASE)


def clamp_citation_indices(text: str, num_sources: int) -> str:
    """
    Drop citation markers that reference a source index outside 1..num_sources
    (e.g. hallucinated [16]). Keeps only in-range numbers inside each bracket.
    """
    if num_sources <= 0:
        return re.sub(r"\s?\[\d+(?:\s*,\s*\d+)*\]", "", text)

    def _repl(match: Match) -> str:
        nums = re.findall(r"\d+", match.group(0))
        valid = [n for n in nums if 1 <= int(n) <= num_sources]
        if not valid:
            return ""
        # Preserve single-number form when possible.
        if len(valid) == 1:
            return f"[{valid[0]}]"
        return "[" + ", ".join(valid) + "]"

    # Only touch numeric citation brackets, not other [text] spans.
    cleaned = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", _repl, text)
    return re.sub(r" {2,}", " ", cleaned).strip()


def renumber_sources_to_citations(
    answer_text: str, sources: List[Dict]
) -> Tuple[str, List[Dict]]:
    """
    After filtering to cited sources only, renumber them 1..k and rewrite
    citations in the answer so returned `n` always matches the bracket used.
    """
    if not sources:
        return answer_text, sources

    old_to_new = {s["n"]: i for i, s in enumerate(sources, 1)}
    # Only rewrite if numbering actually changes.
    if all(old == new for old, new in old_to_new.items()):
        return answer_text, sources

    def _repl(match: Match) -> str:
        nums = re.findall(r"\d+", match.group(0))
        mapped = []
        for n in nums:
            new = old_to_new.get(int(n))
            if new is not None and new not in mapped:
                mapped.append(new)
        if not mapped:
            return match.group(0)
        if len(mapped) == 1:
            return f"[{mapped[0]}]"
        return "[" + ", ".join(str(x) for x in mapped) + "]"

    rewritten = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", _repl, answer_text)
    renumbered = []
    for s in sources:
        s2 = dict(s)
        s2["n"] = old_to_new[s["n"]]
        renumbered.append(s2)
    renumbered.sort(key=lambda s: s["n"])
    return rewritten, renumbered


def group_chunks_by_paper(chunks: List[Dict]) -> "OrderedDict[str, List[Dict]]":
    grouped = OrderedDict()
    for chunk in chunks:
        grouped.setdefault(chunk["arxiv_id"], []).append(chunk)
    return grouped


def build_context_block(
    grouped: "OrderedDict[str, List[Dict]]",
    *,
    abstract_only: bool = False,
    prefer_early_chunks: bool = False,
) -> str:
    lines = []
    for i, (arxiv_id, paper_chunks) in enumerate(grouped.items(), 1):
        title = paper_chunks[0]["title"]
        # Abstract first: it states what the method IS, which is what most
        # questions need — body chunks are often ablation/related-work text
        # that reads badly as a definition. sorted() is stable, so retrieval
        # order is preserved within each source type.
        if abstract_only:
            ordered = [c for c in paper_chunks if c.get("source") == "abstract"]
            if not ordered:
                ordered = sorted(
                    paper_chunks,
                    key=lambda c: (c.get("chunk_index") or 0),
                )[:1]
        else:
            ordered = sorted(
                paper_chunks, key=lambda c: 0 if c.get("source") == "abstract" else 1
            )
            if prefer_early_chunks:
                ordered = sorted(
                    ordered,
                    key=lambda c: (
                        0 if c.get("source") == "abstract" else 1,
                        c.get("chunk_index") or 0,
                    ),
                )
        truncated = []
        for c in ordered[:2]:
            text = (c.get("text") or "").strip()
            if len(text) > MAX_CHUNK_CHARS:
                text = text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + " ..."
            truncated.append(text)
        excerpt_text = "\n---\n".join(truncated)
        lines.append(f"[{i}] Source: {title} ({arxiv_id})\n{excerpt_text}\n")
    return "\n".join(lines)


def _citation_referenced(n: int, answer_text: str) -> bool:
    pattern = r"\[[^\]]*\b" + re.escape(str(n)) + r"\b[^\]]*\]"
    return re.search(pattern, answer_text) is not None


def _source_referenced(source: dict, answer_text: str) -> bool:
    if _citation_referenced(source["n"], answer_text):
        return True
    arxiv_id = source.get("arxiv_id")
    return bool(arxiv_id) and arxiv_id in answer_text


def looks_like_invented_faq(text: str) -> bool:
    question_lines = [
        line.strip() for line in text.split("\n")
        if line.strip().endswith("?") and len(line.strip()) < 150
    ]
    return len(question_lines) >= 2


def ollama_chat_with_deadline(
    model: str,
    messages: List[Dict],
    num_predict: int = OLLAMA_NUM_PREDICT,
    deadline_sec: float = LLM_DEADLINE_SEC,
    node_name: str = "llm",
    format: str = None,
) -> str:
    """
    Streaming Ollama call with a hard wall-clock budget.

    A non-streaming request bounded by `timeout=` only limits the silence
    between received bytes, so a degraded-but-alive generation can run
    arbitrarily long. Streaming lets us watch the clock per token and stop
    at the deadline, keeping whatever was generated so far.
    """
    t0 = time.time()
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "stream": True,
    }
    if format:
        payload["format"] = format
    response = requests.post(
        OLLAMA_URL, json=payload, timeout=(10, 180), stream=True
    )
    response.raise_for_status()

    parts: List[str] = []
    try:
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            parts.append(data.get("message", {}).get("content", ""))
            if data.get("done"):
                break
            if time.time() - t0 > deadline_sec:
                print(
                    f"[{node_name}] LLM call exceeded {deadline_sec:.0f}s "
                    f"deadline — truncating generation",
                    flush=True,
                )
                break
    finally:
        response.close()
    return "".join(parts)


def _call_synth_llm(system_prompt: str, user_prompt: str) -> str:
    return ollama_chat_with_deadline(
        SYNTH_MODEL,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        node_name="synthesizer",
    )


def _query_is_definitional(query: str) -> bool:
    q = query.lower().strip()
    if _query_wants_multi_paper(query):
        return False
    return bool(
        re.match(r"^\s*what\s+(?:is|are)\b", q)
        or re.match(r"^\s*what\s+does\b.+\bmean\b", q)
    )


def _query_asks_typical_value(query: str) -> bool:
    q = query.lower()
    return any(
        w in q
        for w in ("typical", "typically", "usually", "often", "commonly", "standard rank")
    )


def _rank_grouped_by_title(
    grouped: "OrderedDict[str, List[Dict]]", query: str
) -> "OrderedDict[str, List[Dict]]":
    """
    Put the best title match first so [1] is the paper the query most
    clearly names. No paper IDs: uses the same title boost as retrieval.
    """
    from rag.retrieve import _title_boost, _keywords

    kw = _keywords(query)
    scored = []
    for arxiv_id, paper_chunks in grouped.items():
        title = (paper_chunks[0].get("title") or "") if paper_chunks else ""
        avg_dist = 0.5
        if paper_chunks:
            avg_dist = sum((c.get("distance") or 0.5) for c in paper_chunks) / len(
                paper_chunks
            )
        score = _title_boost(title, kw, query) - 0.05 * avg_dist
        scored.append((score, arxiv_id, paper_chunks))
    scored.sort(key=lambda row: row[0], reverse=True)
    return OrderedDict((aid, chunks) for _, aid, chunks in scored)


def _limit_grouped_for_query(
    grouped: "OrderedDict[str, List[Dict]]", query: str
) -> "OrderedDict[str, List[Dict]]":
    """Keep factual answers focused; allow more papers for compare/synthesis."""
    if _query_wants_multi_paper(query):
        max_papers = 3
    else:
        # Single-focus questions (including "what is X?" and "how does X...")
        # should ground on one best title match, not a second near-miss paper.
        max_papers = 1
    if len(grouped) <= max_papers:
        return grouped
    return OrderedDict(list(grouped.items())[:max_papers])


def synthesizer_node(state: dict) -> dict:
    query = state["query"]
    chunks = state.get("retrieved_docs", [])
    notice = state.get("memory_notice")

    if not chunks:
        memory_store = state.get("memory_store")
        is_recall_query = query_requests_memory_recall(query)

        # Paper-read status answers are complete on their own.
        if notice and query_asks_if_paper_read(query):
            return {
                "answer": notice,
                "sources": [],
                "tool_calls": state.get("tool_calls", []) + ["synthesizer(paper_read_status)"],
            }

        # Unclear/gibberish notice with no docs — return the clarification, don't
        # invent an answer.
        if notice and (
            "couldn't identify a clear research question" in notice.lower()
            or "own implementation" in notice.lower()
        ):
            return {
                "answer": notice,
                "sources": [],
                "tool_calls": state.get("tool_calls", []) + ["synthesizer(notice_only)"],
            }

        if is_recall_query and memory_store and memory_store._current_session_id:
            topics_list = memory_store.get_session_topics(memory_store._current_session_id)
            conclusions_list = memory_store.get_session_conclusions(memory_store._current_session_id)

            if topics_list or conclusions_list:
                topics = ", ".join(topics_list) or "none recorded"
                conclusions = "\n".join(f"- {c}" for c in conclusions_list) or "none recorded"
                answer = f"So far we've covered: {topics}.\n\nKey points so far:\n{conclusions}"
                if notice:
                    answer = f"{notice}\n\n{answer}"
                return {
                    "answer": answer,
                    "sources": [],
                    "tool_calls": state.get("tool_calls", []) + ["synthesizer(from_memory)"],
                }

        if is_recall_query:
            answer = "We haven't discussed anything yet in this session."
        else:
            answer = "I couldn't find any relevant information for this question, either locally or via search."
        if notice:
            answer = f"{notice}\n\n{answer}"
        return {
            "answer": answer,
            "sources": [],
            "tool_calls": state.get("tool_calls", []) + ["synthesizer(no_docs)"],
        }

    # If we somehow retrieved docs for a gibberish/unclear query, refuse to
    # synthesize a fabricated comparison from them.
    if notice and "couldn't identify a clear research question" in notice.lower():
        return {
            "answer": notice,
            "sources": [],
            "tool_calls": state.get("tool_calls", []) + ["synthesizer(refuse_unclear)"],
        }

    grouped = group_chunks_by_paper(chunks)
    grouped = _rank_grouped_by_title(grouped, query)
    grouped = _limit_grouped_for_query(grouped, query)

    queried_id = state.get("_queried_arxiv_id")
    if queried_id:
        returned_ids = set(grouped.keys())
        if queried_id not in returned_ids:
            answer = (
                f"I could not find paper {queried_id} in my local database or via arXiv search. "
                f"Either the paper ID does not exist or it hasn't been ingested yet."
            )
            if notice:
                answer = f"{notice}\n\n{answer}"
            return {
                "answer": answer,
                "sources": [],
                "tool_calls": state.get("tool_calls", []) + ["synthesizer(paper_not_found)"],
            }

    context_block = build_context_block(
        grouped,
        abstract_only=_query_is_definitional(query),
        prefer_early_chunks=_query_asks_typical_value(query),
    )
    n_sources = len(grouped)

    if _query_wants_multi_paper(query):
        length_hint = (
            "Write a focused comparative answer (one short section), citing the "
            "excerpt numbers shown (e.g. [1], [2]). Cite each paper whose "
            "claims you actually use. Only cite numbers that appear on the "
            "excerpts above — never invent indices."
        )
    else:
        length_hint = (
            f"Write 2-4 concise sentences answering ONLY the question. Choose "
            f"the best-supported excerpt among the {n_sources} source(s), cite "
            f"its actual number, and never write the literal text [n]. Only use "
            f"citation numbers in 1..{n_sources}."
        )
    if _query_asks_typical_value(query):
        length_hint += (
            " When the question asks for a typical/usual/common value, report "
            "the range the authors present as standard practice (e.g. ranks "
            "commonly used in experiments). Do NOT treat a minimum ablation "
            "rank that 'still works' or 'is already competitive' as the typical "
            "value — those are edge cases, not the usual setting."
        )
    if _query_is_definitional(query):
        length_hint += (
            " Prefer the abstract-level definition (what the method is and why "
            "it exists) over peripheral body details."
        )

    user_prompt = f"""Question: {query}

Excerpts:
{context_block}

{length_hint}
Do not add headers, FAQ-style sub-questions, or sections."""

    try:
        raw = _call_synth_llm(SYSTEM_PROMPT, user_prompt)
        answer_text = sanitize_answer(raw)
        answer_text = fix_literal_n_citations(answer_text, n_sources)
        answer_text = clamp_citation_indices(answer_text, n_sources)

        if looks_like_invented_faq(answer_text):
            print("[synthesizer] detected invented-FAQ pattern, retrying with stricter prompt")
            retry_prompt = f"""Question: {query}

Excerpts:
{context_block}

Write ONE short paragraph, 3-5 sentences, that answers ONLY: "{query}"
Do not write any other question. Do not use headers. Do not use bullet points.
Cite sources with their real excerpt number in brackets (e.g. [1], [2]).
Never write the literal text [n]. Only use citation numbers in 1..{n_sources}.
Begin your response with the answer itself, nothing else."""
            raw_retry = _call_synth_llm(SYSTEM_PROMPT, retry_prompt)
            retry_answer = clamp_citation_indices(
                fix_literal_n_citations(sanitize_answer(raw_retry), n_sources),
                n_sources,
            )
            if not looks_like_invented_faq(retry_answer):
                answer_text = retry_answer

    except (requests.ConnectionError, requests.Timeout, requests.HTTPError, KeyError) as e:
        answer_text = f"[ERROR: could not reach local Ollama server — {e}]"

    if notice:
        answer_text = f"{notice}\n\n{answer_text}" if answer_text else notice

    all_sources = [
        {
            "n": i,
            "title": paper_chunks[0].get("title", ""),
            "url": paper_chunks[0].get("url", ""),
            "arxiv_id": arxiv_id,
        }
        for i, (arxiv_id, paper_chunks) in enumerate(grouped.items(), 1)
    ]

    sources = [s for s in all_sources if _source_referenced(s, answer_text)]
    # If the model cited nothing usable, keep the top source so the API still
    # surfaces what grounded the answer — but only after clamp left no cites.
    if not sources and all_sources and answer_text and not answer_text.startswith("[ERROR"):
        sources = all_sources[:1]
        if not re.search(r"\[\d+\]", answer_text):
            answer_text = answer_text.rstrip() + " [1]"

    answer_text, sources = renumber_sources_to_citations(answer_text, sources)

    return {
        "answer": answer_text,
        "sources": sources,
        "tool_calls": state.get("tool_calls", []) + [f"synthesizer({len(grouped)} papers)"],
    }
