import json
import re
import requests

from memory.context_injection import format_memory_context
from memory.routing import (
    decide_next_action,
    query_requests_memory_recall,
    extract_arxiv_id,
    query_asks_if_paper_read,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
PLANNER_MODEL = "llama3.1:8b"

PLANNER_SYSTEM_PROMPT = """You are a routing planner for a research assistant agent.
Given the user's query and what has already been retrieved, decide the single next action.

Return ONLY a JSON object, no other text, in this exact shape:
{"action": "search_arxiv" | "retrieve_local" | "answer_now" | "compare_papers", "reason": "<one short sentence>", "query_understood": true | false}

Rules:
- Prefer "retrieve_local" first when the topic is likely already in the local corpus
  (e.g. LoRA, RAG, federated fine-tuning, PEFT). Only use "search_arxiv" when the
  query explicitly asks for recent/new papers, or the topic is clearly absent locally.
- Use "search_arxiv" if the query needs papers we don't have yet (new topic, no local docs retrieved).
- Use "retrieve_local" if we likely already have relevant papers ingested locally and haven't checked yet.
- Use "compare_papers" ONLY if the query explicitly asks to compare, contrast, or evaluate differences
  between multiple papers/approaches, AND retrieved_docs already contains 2 or more distinct papers.
- Use "answer_now" if we already have enough retrieved_docs or search_results to answer, or the query needs no external info.
- If search_results is already populated (has_search_results: true), do NOT choose "search_arxiv" again — choose "answer_now" or "retrieve_local" instead.
- If retrieved_docs is already populated (has_retrieved_docs: true), do NOT choose "retrieve_local" again.
- If the query asks about this system's own implementation, model, or architecture rather than about paper content, use "answer_now" — do not search for papers.
- Set "query_understood" to false ONLY if the query is not a coherent, parseable request at all —
  e.g. random keyboard mashing, no discernible topic or intent. The query does NOT need to be
  grammatically correct or well-formed to count as understood — if there is ANY discernible research
  topic or intent, even a rough or malformed one, set "query_understood" to true. When
  "query_understood" is false, set "action" to "answer_now" regardless of any other rule above —
  never search or compare against a query you can't actually parse.
"""

VALID_ACTIONS = {"search_arxiv", "retrieve_local", "answer_now", "compare_papers"}

COMPARE_KEYWORDS = {
    "compare", "comparison", "versus", " vs ", " vs. ",
    "contrast", "difference", "which is better",
}
_COMPARE_WORD_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|contrast|difference|differences|differ)\b"
    r"|\bwhich\s+is\s+better\b",
    re.IGNORECASE,
)

# "Do papers agree on how X compares to Y?" is multi-paper synthesis, not a
# structured A-vs-B compare. Substring "compare" inside "compares" used to
# force the heavier comparator node and blow up latency.
_SYNTHESIS_AGREE_RE = re.compile(
    r"\b(?:do\s+(?:different\s+|the\s+)?papers\s+agree|agree\s+on\s+how|"
    r"across\s+papers|different\s+papers\s+agree)\b",
    re.IGNORECASE,
)

# Action/filler words stripped before gibberish topic detection.
_ACTION_NOISE = {
    "compare", "comparison", "contrast", "versus", "vs", "difference", "differ",
    "differences", "papers", "paper", "find", "search", "recent", "about",
    "the", "a", "an", "what", "is", "are", "how", "does", "do", "which",
    "better", "and", "or", "to", "of", "in", "on", "for", "with", "please",
    "tell", "me", "can", "you", "i", "we", "them", "their", "this", "that",
    "then", "also", "just", "any", "some", "top", "two", "between",
}

# Explicit signals that the user wants a fresh arXiv search, not local RAG.
_FORCE_SEARCH_KEYWORDS = {
    "find recent", "recent papers", "newest papers", "latest papers",
    "search arxiv", "search arxiv for", "new papers on",
}


SELF_KNOWLEDGE_KEYWORDS = {
    "this system", "your embedding", "your model", "you use", "do you use",
    "what model are you", "which model are you", "what embedding", "which embedding",
    "your architecture", "your implementation", "your database",
}


def query_requests_comparison(query: str) -> bool:
    q = query.lower()
    if _SYNTHESIS_AGREE_RE.search(q):
        return False
    # Word-boundary match so "different layers" does not force compare_papers.
    return bool(_COMPARE_WORD_RE.search(q)) or any(
        kw in q for kw in COMPARE_KEYWORDS if kw not in ("differ",)
    )


def query_is_self_referential(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SELF_KNOWLEDGE_KEYWORDS)


def query_explicitly_wants_search(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _FORCE_SEARCH_KEYWORDS)


def _token_looks_real(token: str) -> bool:
    if len(token) < 3:
        return False
    vowels = sum(1 for c in token if c in "aeiou")
    if vowels / len(token) < 0.2:
        return False
    # Reject near-duplicate-char keyboard smash (e.g. "aaaaaa")
    if len(set(token)) < min(3, len(token)):
        return False
    return True


def query_looks_gibberish(query: str) -> bool:
    """
    Deterministic check for keyboard-mashing / no-topic queries.
    Strips action words like "compare papers" so leftover topic tokens
    must look like real words — otherwise we refuse to search.
    """
    tokens = re.findall(r"[a-zA-Z]{2,}", query.lower())
    if not tokens:
        return True
    topic_tokens = [t for t in tokens if t not in _ACTION_NOISE]
    if not topic_tokens:
        return True
    real = [t for t in topic_tokens if _token_looks_real(t)]
    return len(real) == 0


def _get_memory_context(state: dict) -> str | None:
    if state.get("memory_context"):
        return state["memory_context"]

    memory_store = state.get("memory_store")
    if memory_store is None:
        return None

    snapshot = memory_store.get_memory_snapshot()
    return format_memory_context(snapshot)


def _resolve_paper_read_notice(memory_store, routing_decision: dict) -> str:
    """Build a yes/no answer for 'have you read the X paper?' queries."""
    arxiv_id = routing_decision.get("arxiv_id")
    keyword = routing_decision.get("title_keyword")

    if arxiv_id:
        paper = memory_store.get_paper(arxiv_id)
        if paper is not None:
            return (
                f"Yes — I've already read '{paper.title}' ({paper.arxiv_id}). "
                f"I can reuse the cached copy instead of re-fetching it."
            )
        return (
            f"No — I have not read paper {arxiv_id} yet "
            f"(it is not in my papers_read cache)."
        )

    if keyword and not re.fullmatch(r"\d{4}\.\d{4,5}", keyword):
        matches = memory_store.find_papers_by_title_keyword(keyword, limit=3)
        if matches:
            # find_papers_by_title_keyword ranks canonical titles first
            # (e.g. "LoRA: Low-Rank Adaptation..."), so lead with the best
            # match instead of dumping every title that mentions the keyword.
            # Single best match only — listing every title that mentions the
            # keyword confuses "have you read the X paper?" with a survey.
            best = matches[0]
            return (
                f"Yes — I've already read '{best.title}' ({best.arxiv_id}). "
                f"I can reuse the cached copy instead of re-fetching it."
            )
        return (
            f"No — I don't have a paper matching '{keyword}' in my papers_read cache yet."
        )

    known_ids = memory_store.get_memory_snapshot().known_paper_ids
    if known_ids:
        return (
            f"I've read {len(known_ids)} paper(s) before, but I couldn't tell which "
            f"specific paper you're asking about. Try naming it or giving an arXiv ID."
        )
    return "No — I haven't read any papers yet in my cache."


def planner_node(state: dict) -> dict:
    hop_count = state.get("hop_count", 0)

    queried_arxiv_id = state.get("_queried_arxiv_id")
    memory_store = state.get("memory_store")

    if hop_count == 0 and queried_arxiv_id is None:
        candidate = extract_arxiv_id(state["query"])
        if candidate and memory_store is not None:
            known_paper = memory_store.get_paper(candidate)
            if known_paper is None:
                queried_arxiv_id = candidate
        elif candidate:
            # No memory store — still track the ID so synthesizer can refuse
            # if search returns unrelated papers for a fake ID.
            queried_arxiv_id = candidate

    if hop_count >= 4:  # safety valve
        return {
            "next_action": "answer_now",
            "hop_count": hop_count + 1,
            "tool_calls": state.get("tool_calls", []) + ["planner:forced_answer_now(max_hops)"],
            "_queried_arxiv_id": queried_arxiv_id,
        }

    retrieved_docs = state.get("retrieved_docs", [])
    query = state["query"]
    memory_context = _get_memory_context(state)

    # --- Early exits before calling the LLM planner ---

    # Hard gibberish gate — do not trust the LLM's query_understood flag alone.
    if hop_count == 0 and query_looks_gibberish(query):
        return {
            "next_action": "answer_now",
            "hop_count": hop_count + 1,
            "tool_calls": state.get("tool_calls", []) + ["planner:answer_now(gibberish)"],
            "memory_notice": (
                "I couldn't identify a clear research question in that message — "
                "it doesn't seem to reference a coherent topic. Could you rephrase it?"
            ),
            "retrieved_docs": [],
            "_queried_arxiv_id": queried_arxiv_id,
        }

    if hop_count == 0 and query_is_self_referential(query):
        return {
            "next_action": "answer_now",
            "hop_count": hop_count + 1,
            "tool_calls": state.get("tool_calls", []) + ["planner:answer_now(self_referential)"],
            "memory_notice": (
                "This question is about the system's own implementation, not about paper content. "
                "I can only answer questions grounded in the research papers I have access to."
            ),
            "_queried_arxiv_id": queried_arxiv_id,
        }

    if memory_store is not None and hop_count == 0:
        snapshot = memory_store.get_memory_snapshot()
        routing_decision = decide_next_action(query, snapshot)

        if routing_decision["action"] == "report_paper_read":
            notice = _resolve_paper_read_notice(memory_store, routing_decision)
            return {
                "next_action": "answer_now",
                "hop_count": hop_count + 1,
                "tool_calls": state.get("tool_calls", []) + ["planner:answer_now(paper_read_status)"],
                "memory_notice": notice,
                "retrieved_docs": [],
                "_queried_arxiv_id": queried_arxiv_id,
            }

        if routing_decision["action"] == "use_cached_paper":
            cached = memory_store.get_paper(routing_decision["arxiv_id"])
            if cached is not None:
                return {
                    "next_action": "retrieve_local",
                    "hop_count": hop_count + 1,
                    "tool_calls": state.get("tool_calls", []) + [f"planner:use_cached_paper({cached.arxiv_id})"],
                    "memory_notice": routing_decision["notice"],
                    "retrieval_filter": {"arxiv_id": cached.arxiv_id},
                    "retrieved_docs": [{
                        "arxiv_id": cached.arxiv_id,
                        "title": cached.title,
                        "url": cached.url or "",
                        "text": cached.abstract or f"Cached paper {cached.arxiv_id}",
                        "source": "memory",
                        "chunk_index": 0,
                        "distance": 0.0,
                    }],
                    "_queried_arxiv_id": queried_arxiv_id,
                }

    # Prefer local retrieve on the first hop unless the user explicitly asked
    # for a fresh/recent search. Avoids 200s+ arXiv round-trips for topics
    # already in the corpus.
    if (
        hop_count == 0
        and not retrieved_docs
        and not query_explicitly_wants_search(query)
        and not query_requests_memory_recall(query)
        and not query_asks_if_paper_read(query)
    ):
        return {
            "next_action": "retrieve_local",
            "hop_count": hop_count + 1,
            "tool_calls": state.get("tool_calls", []) + ["planner:retrieve_local(prefer_local)"],
            "_queried_arxiv_id": queried_arxiv_id,
        }

    try:
        distinct_papers = len(
            set(d.get("arxiv_id") for d in retrieved_docs if d.get("arxiv_id"))
        ) if retrieved_docs else 0

        user_context = {
            "query": query,
            "has_retrieved_docs": bool(retrieved_docs),
            "distinct_papers_retrieved": distinct_papers,
            "has_search_results": bool(state.get("search_results")),
        }

        system_prompt = PLANNER_SYSTEM_PROMPT
        if memory_context:
            system_prompt = f"{system_prompt}\n\n{memory_context}"

        from agents.synthesizer import ollama_chat_with_deadline
        raw = ollama_chat_with_deadline(
            PLANNER_MODEL,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_context)},
            ],
            num_predict=200,   # routing decision is a tiny JSON object
            deadline_sec=120,
            node_name="planner",
            format="json",
        )
        parsed = json.loads(raw)
        action = parsed.get("action")
        query_understood = parsed.get("query_understood", True)
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action from planner: {action}")

        unclear_query_notice = None
        if not query_understood or query_looks_gibberish(query):
            action = "answer_now"
            unclear_query_notice = (
                "I couldn't identify a clear research question in that message — "
                "it doesn't seem to reference a coherent topic. Could you rephrase it?"
            )

        if query_requests_memory_recall(query):
            action = "answer_now"

        if action == "compare_papers" and not query_requests_comparison(query):
            action = "answer_now"

        # Need 2+ distinct papers before compare is allowed.
        if action == "compare_papers" and distinct_papers < 2:
            action = "retrieve_local" if not retrieved_docs else "search_arxiv"

        # If the user asked to compare and we already have 2+ papers, force compare
        # rather than narrating a single-paper answer_now.
        if (
            query_requests_comparison(query)
            and distinct_papers >= 2
            and action in ("answer_now", "retrieve_local")
        ):
            action = "compare_papers"

        if action == "search_arxiv" and state.get("search_results"):
            action = "answer_now"

        if action == "retrieve_local" and retrieved_docs:
            if query_requests_comparison(query) and distinct_papers >= 2:
                action = "compare_papers"
            else:
                action = "answer_now"

        # After a local retrieve that returned nothing useful, allow one search
        # hop — but only if the user asked for recent/new material OR local
        # docs are empty. Otherwise answer with what we have.
        if action == "search_arxiv" and retrieved_docs and not query_explicitly_wants_search(query):
            action = "answer_now" if distinct_papers >= 1 else action

    except (requests.ConnectionError, requests.Timeout, requests.HTTPError,
            json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[planner] error: {e}")
        action = "answer_now"
        unclear_query_notice = None

    return {
        "next_action": action,
        "hop_count": hop_count + 1,
        "tool_calls": state.get("tool_calls", []) + [f"planner:{action}"],
        "memory_notice": unclear_query_notice or state.get("memory_notice"),
        "_queried_arxiv_id": queried_arxiv_id,
    }
