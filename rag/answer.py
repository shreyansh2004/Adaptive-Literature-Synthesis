from typing import List, Dict
from dataclasses import dataclass
from collections import OrderedDict

from rag.retrieve import retrieve, RetrievedChunk
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.1:8b" 

def _call_llm_with_retry(messages, max_retries=2, base_wait=5):
    """
    Call local Ollama with basic retry.
    Local failures are usually transient (server still starting, model still
    loading into memory) rather than rate limits, so short waits are enough.
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": LLM_MODEL,
                "messages": messages,
                "options": {"temperature": 0},
                "stream": False,
            }, timeout=300)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == max_retries:
                raise
            wait = base_wait * (attempt + 1)
            print(f"[ollama] retrying in {wait}s... ({e})")
            time.sleep(wait)


def sanitize_answer(text: str) -> str:
    """
    Remove leading equations / notation before the actual prose answer.
    """
    lines = text.strip().splitlines()

    while lines:
        line = lines[0].strip()

        if not line:
            lines.pop(0)
            continue

        alpha = sum(c.isalpha() for c in line)

        # Looks like prose
        if alpha / max(len(line), 1) > 0.45:
            break

        # Otherwise discard it
        lines.pop(0)

    return "\n".join(lines).strip()

SYSTEM_PROMPT = """You are a research assistant answering questions using only the provided paper excerpts.

Rules:
- Only use information from the provided excerpts. Do not use outside knowledge.
- If the excerpts don't contain enough information to answer, say so explicitly.
- Cite sources inline using [n] notation matching the excerpt numbers given.
- Excerpts come from DIFFERENT papers. Never merge findings from different papers into
  a single claim unless the question explicitly asks for a comparison.
- Your answer must be written entirely in your own words. Do not include any raw text,
  equations, or notation copied directly from the excerpts — describe formulas and results
  in plain language instead of reproducing their symbols.
- Do not begin your answer with equations, symbols, or excerpt fragments. Start directly
  with a plain-language sentence answering the question.
- Be concise and precise. Do not pad the answer with filler.
"""


@dataclass
class AnswerResult:
    answer: str
    sources: List[Dict]


def group_chunks_by_paper(chunks: List[RetrievedChunk]) -> "OrderedDict[str, List[RetrievedChunk]]":
    """
    Group retrieved chunks by arxiv_id, preserving the order in which each
    paper first appeared (i.e. by best relevance rank).
    """
    grouped = OrderedDict()
    for chunk in chunks:
        grouped.setdefault(chunk.arxiv_id, []).append(chunk)
    return grouped


def build_context_block(grouped: "OrderedDict[str, List[RetrievedChunk]]") -> str:
    """One numbered entry per PAPER, with its matched excerpt(s) concatenated underneath."""
    lines = []
    for i, (arxiv_id, paper_chunks) in enumerate(grouped.items(), 1):
        title = paper_chunks[0].title
        lines.append(f"[{i}] Source: {title} ({arxiv_id})")
        for chunk in paper_chunks:
            lines.append(f"[{i}] {chunk.text}")
        lines.append("")
    return "\n".join(lines)


def answer_question(query: str, top_k: int = 6) -> AnswerResult:
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return AnswerResult(
            answer="I couldn't find any relevant information in the local paper collection.",
            sources=[],
        )

    grouped = group_chunks_by_paper(chunks)
    context_block = build_context_block(grouped)

    user_prompt = f"""Question: {query}

Excerpts:
{context_block}

Answer the question using only the excerpts above. Cite using [n]."""

    try:
        response = _call_llm_with_retry([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        answer_text = sanitize_answer(response["message"]["content"])

    except Exception as e:
        answer_text = f"[ERROR: rate limit hit, could not generate answer — {e}]"


    sources = [
        {"n": i, "title": paper_chunks[0].title, "url": paper_chunks[0].url, "arxiv_id": arxiv_id}
        for i, (arxiv_id, paper_chunks) in enumerate(grouped.items(), 1)
    ]

    return AnswerResult(answer=answer_text, sources=sources)


if __name__ == "__main__":
    query = "How does LoRA reduce the number of trainable parameters?"
    result = answer_question(query)

    print(f"Q: {query}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for s in result.sources:
        print(f"  [{s['n']}] {s['title']} — {s['url']}")