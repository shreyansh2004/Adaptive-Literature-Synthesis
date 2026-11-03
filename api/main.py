"""
FastAPI wrapper for the LangGraph research assistant agent.

Day 22 milestone: a /chat endpoint that runs a query through the agent
graph, with session-ID-based memory so repeat calls with the same
session_id build on prior context. Test via curl/Postman (examples below).

Run:
    uvicorn api.main:api --reload --port 8000

Example curl:
    # First turn — no session_id, server creates one
    curl -X POST http://localhost:8000/chat \\
      -H "Content-Type: application/json" \\
      -d '{"query": "What is LoRA fine-tuning?"}'

    # Follow-up — reuse the session_id from the first response
    curl -X POST http://localhost:8000/chat \\
      -H "Content-Type: application/json" \\
      -d '{"query": "What did we just discuss?", "session_id": "<id from above>"}'
"""

import os
import time
import uuid
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.graph import app as compiled_graph_app
from memory.session_summarizer import summarize_and_persist
from memory.store import MemoryStore

MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", "data/memory.db")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "llama3.1:8b")

api = FastAPI(title="Research Assistant API")


@api.on_event("startup")
def startup_checks():
    os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[dict] = []          # matches synthesizer: [{"n", "title", "url", "arxiv_id"}]
    memory_notice: Optional[str] = None
    tool_calls: List[str] = []


def _make_initial_state(query: str, memory_store: MemoryStore) -> dict:
    """Mirrors the AgentState schema used by planner_node / the graph."""
    return {
        "query": query,
        "retrieved_docs": [],
        "search_results": [],
        "tool_calls": [],
        "answer": None,
        "sources": [],
        "next_action": None,
        "hop_count": 0,
        "memory_store": memory_store,
        "memory_context": None,
        "memory_notice": None,
        "retrieval_filter": None,
        "_queried_arxiv_id": None,
    }


def _summarizer_llm(prompt: str) -> str:
    """
    LLM callable passed to summarize_and_persist. Expected to return a
    JSON string of shape {"summary": str, "topics": [str],
    "conclusions": [{"text": str, "topic": str, "arxiv_id": str|None}]}
    — same schema the Day 16 tests exercise via a fake_llm stub.
    """
    from agents.synthesizer import ollama_chat_with_deadline
    return ollama_chat_with_deadline(
        SUMMARIZER_MODEL,
        [{"role": "user", "content": prompt}],
        num_predict=500,
        deadline_sec=150,
        node_name="summarizer",
        format="json",
    )


from agents.planner import query_requests_memory_recall

def _run_summarize_and_persist(memory_store: MemoryStore, turns: List[dict]):
    t0 = time.time()
    try:
        # summarize_and_persist calls store.end_session() internally — don't call it again.
        result = summarize_and_persist(memory_store, turns, _summarizer_llm)
        print(f"[timing] summarizer: {time.time() - t0:.1f}s", flush=True)
        print(f"[chat] summarize result: {result}", flush=True)  
    except Exception as e:
        print(f"[chat] summarize_and_persist failed after {time.time() - t0:.1f}s: {e}", flush=True)
        memory_store.end_session(summary=turns[0]["answer"][:500])


@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    session_id = req.session_id or str(uuid.uuid4())

    memory_store = MemoryStore(db_path=MEMORY_DB_PATH)
    memory_store.start_session(session_id)

    try:
        initial_state = _make_initial_state(req.query, memory_store)
        final_state = compiled_graph_app.invoke(initial_state)
    except Exception as e:
        # Close the session before surfacing the error so it doesn't linger open.
        memory_store.end_session(summary=f"Session ended after agent error: {e}")
        raise HTTPException(status_code=500, detail=f"agent run failed: {e}")

    answer = final_state.get("answer") or ""
    retrieved_docs = final_state.get("retrieved_docs", [])
    
    turns = [{
        "query": req.query,
        "answer": answer,
        "arxiv_ids": [
            d.get("arxiv_id") for d in retrieved_docs
            if d.get("arxiv_id")
        ],
    }]

    # Skip the expensive LLM summarization if no new papers were retrieved 
    # or if the query is just recalling past memory.
    if query_requests_memory_recall(req.query) or not retrieved_docs:
        with memory_store._conn() as conn:
            existing = conn.execute(
                "SELECT summary FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not existing or not existing["summary"]:
            memory_store.end_session(summary="No new papers discussed yet.")
        else:
            memory_store.end_session(summary=existing["summary"])  # preserve it
        print("[chat] skipped summarization (no new papers discussed)", flush=True)
    else:
        _run_summarize_and_persist(memory_store, turns)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=final_state.get("sources", []),
        memory_notice=final_state.get("memory_notice"),
        tool_calls=final_state.get("tool_calls", []),
    )


@api.get("/health")
def health():
    return {"status": "ok"}


@api.get("/sessions/{session_id}")
def get_session(session_id: str):
    store = MemoryStore(db_path=MEMORY_DB_PATH)
    snapshot = store.get_memory_snapshot()
    return {
        "recent_topics": snapshot.recent_topics,
        "recent_conclusions": snapshot.recent_conclusions,
        "known_paper_ids": list(snapshot.known_paper_ids),
        "preferences": snapshot.preferences,
    }