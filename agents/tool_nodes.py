"""
Tool nodes: thin wrappers around Week 1 functions, adapted to the
(state in) -> (state update out) shape LangGraph nodes require.
"""

from dataclasses import asdict
import time

from tools.arxiv_mcp_client import search_arxiv_via_mcp
# from tools.arxiv_search import search_arxiv
from rag.retrieve import retrieve
from tools.paper_fetch import fetch_and_save, load_paper
from rag.ingest import ingest_paper
from rag.store import VectorStore

MAX_PAPERS_TO_INGEST = 3  # cap per search — avoid fetching all 5 results' full PDFs every hop


def search_arxiv_node(state: dict) -> dict:
    """Calls the arxiv API, stores structured results in state."""
    query = state["query"]
    try:
        papers = search_arxiv_via_mcp(query, max_results=5)
        results = papers  # papers are already dicts from MCP client
        error = None
    except BaseException as e:
        print(f"[search_arxiv_node] error: {e}")
        results = []
        error = str(e)

    return {
        "search_results": results,
        "tool_calls": state.get("tool_calls", []) + [
            f"search_arxiv({'error: ' + error if error else len(results)})"
        ],
    }


def retrieve_local_node(state: dict) -> dict:
    """Retrieves top-k chunks from the local vector store."""
    query = state["query"]
    where = state.get("retrieval_filter")
    memory_store = state.get("memory_store")
    try:
        chunks = retrieve(query, top_k=6, where=where)
        # RetrievedChunk is a dataclass in rag/retrieve.py — convert to plain dicts
        docs = [asdict(c) for c in chunks]
        error = None
    except Exception as e:
        print(f"[retrieve_local_node] error: {e}")
        docs = state.get("retrieved_docs", [])  # preserve prior docs (e.g. cached abstract stub)
        error = str(e)

    # Remember papers we actually used so "have you read X?" can answer from cache
    # even when we never went through search/fetch.
    if memory_store is not None and docs:
        seen = set()
        for d in docs:
            arxiv_id = d.get("arxiv_id")
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            try:
                if not memory_store.has_read_paper(arxiv_id):
                    memory_store.record_paper_read(
                        arxiv_id=arxiv_id,
                        title=d.get("title") or arxiv_id,
                        url=d.get("url"),
                        abstract=(d.get("text") or "")[:500],
                    )
            except Exception as e:
                print(f"[retrieve_local_node] memory record failed for {arxiv_id}: {e}")

    return {
        "retrieved_docs": docs,
        "tool_calls": state.get("tool_calls", []) + [
            f"retrieve_local({'error: ' + error if error else len(docs)})"
        ],
        "retrieval_filter": None,
    }

def fetch_and_ingest_node(state: dict) -> dict:
    """
    Takes arxiv_ids from search_results, fetches full text (or hits cache),
    and ingests each into the vector store so retrieve_local_node can find them.
    """
    search_results = state.get("search_results", [])
    store = VectorStore()
    memory_store = state.get("memory_store")

    newly_ingested = 0
    already_cached = 0
    failed = 0
    total_chunks = 0

    for i, paper_meta in enumerate(search_results[:MAX_PAPERS_TO_INGEST]):
        if i > 0:
            time.sleep(3)  # be polite to arXiv's API
        arxiv_id = paper_meta["arxiv_id"]
        try:
            already_had_paper = load_paper(arxiv_id) is not None
            already_had_chunks = store.paper_already_ingested(arxiv_id)

            full_text_path = fetch_and_save(arxiv_id, include_full_text=True)  # no-op if already cached
            paper = load_paper(arxiv_id)
            if paper is None:
                raise ValueError(f"fetch_and_save reported success but load_paper returned None for {arxiv_id}")
            
            n_chunks = ingest_paper(paper.to_dict(), store)
            total_chunks += n_chunks

            if memory_store is not None:
                memory_store.record_paper_read(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    url=paper.url,
                    abstract=paper.abstract,
                    full_text_path=full_text_path,
                )
            
            if already_had_paper and already_had_chunks:
                already_cached += 1
            else:
                newly_ingested += 1

        except Exception as e:
            print(f"[fetch_and_ingest_node] error for {arxiv_id}: {e}")
            failed += 1

    return {
        "tool_calls": state.get("tool_calls", []) + [
            f"fetch_and_ingest(new={newly_ingested}, cached={already_cached}, "
            f"failed={failed}, chunks_added={total_chunks})"
        ],
    }