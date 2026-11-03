import time
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from agents.planner import planner_node
from agents.tool_nodes import search_arxiv_node, retrieve_local_node, fetch_and_ingest_node
from agents.synthesizer import synthesizer_node
from agents.comparator import compare_papers_node


def _timed(name, fn):
    """Log per-node wall time so latency spikes are attributable to a node."""
    def wrapper(state):
        t0 = time.time()
        result = fn(state)
        print(f"[timing] {name}: {time.time() - t0:.1f}s", flush=True)
        return result
    return wrapper


class AgentState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    tool_calls: List[str]
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    next_action: Optional[str]
    hop_count: int
    memory_store: Optional[Any]
    memory_context: Optional[str]
    memory_notice: Optional[str]
    retrieval_filter: Optional[Dict[str, Any]]
    _queried_arxiv_id: Optional[str]


def route(state: AgentState) -> str:
    return state["next_action"]


graph = StateGraph(AgentState)
graph.add_node("planner", _timed("planner", planner_node))
graph.add_node("search_arxiv", _timed("search_arxiv", search_arxiv_node))
graph.add_node("fetch_and_ingest", _timed("fetch_and_ingest", fetch_and_ingest_node))
graph.add_node("retrieve_local", _timed("retrieve_local", retrieve_local_node))
graph.add_node("answer", _timed("synthesizer", synthesizer_node))
graph.add_node("compare_papers", _timed("comparator", compare_papers_node))

graph.set_entry_point("planner")
graph.add_conditional_edges(
    "planner",
    route,
    {
        "answer_now": "answer",
        "search_arxiv": "search_arxiv",   
        "retrieve_local": "retrieve_local", 
        "compare_papers": "compare_papers", 
    }
)
# search_arxiv is now a fixed pipeline: search -> fetch+ingest -> retrieve -> back to planner
graph.add_edge("search_arxiv", "fetch_and_ingest")
graph.add_edge("fetch_and_ingest", "retrieve_local")
graph.add_edge("retrieve_local", "planner")   # shared exit point for both entry paths
graph.add_edge("compare_papers", END)  # terminal, like "answer"
graph.add_edge("answer", END)

app = graph.compile()
# synthesizer_node only reads state['retrieved_docs'], not state['search_results'] 
# by design, since the search_arxiv path always ingests+retrieves before reaching answer_now

if __name__ == "__main__":
    test_queries = [
        {"query": "What is LoRA fine-tuning?", "retrieved_docs": [], "search_results": []},
        {"query": "Summarize what we just found", "retrieved_docs": [], "search_results": [{"dummy": "result"}]},
        {
            "query": "What did that paper say about overfitting?",
            "retrieved_docs": [{
                "text": "Overfitting occurs when a model memorizes training data rather than generalizing to new examples.",
                "arxiv_id": "9999.99999",
                "title": "Fake Test Paper",
                "authors": "Test Author",
                "published": "2024-01-01",
                "updated": "2024-01-01",
                "url": "http://arxiv.org/abs/9999.99999",
                "chunk_index": 0,
                "source": "abstract",
                "distance": 0.1,
            }],
            "search_results": [],
        },
        {
            "query": "Compare LoRA-FAIR and Federated Sketching LoRA — how does each reduce cost or improve aggregation?",
            "retrieved_docs": [
                {
                    "text": "LoRA-FAIR introduces a server-side aggregation bias correction term combined with a client-side initialization refinement step to address inconsistencies that arise when aggregating LoRA updates across heterogeneous federated clients.",
                    "arxiv_id": "2411.14961",
                    "title": "LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement",
                    "authors": "Test Author",
                    "published": "2024-11-01",
                    "updated": "2024-11-01",
                    "url": "http://arxiv.org/abs/2411.14961v3",
                    "chunk_index": 0,
                    "source": "abstract",
                    "distance": 0.1,
                },
                {
                    "text": "Federated Sketching LoRA (FSLoRA) uses a sketching mechanism to compress LoRA updates before transmission, reducing communication overhead as well as gradient computation and update costs compared to standard federated LoRA fine-tuning approaches.",
                    "arxiv_id": "2501.19389",
                    "title": "Federated Sketching LoRA: A Flexible Framework for Heterogeneous Collaborative Fine-Tuning of LLMs",
                    "authors": "Test Author",
                    "published": "2025-01-01",
                    "updated": "2025-01-01",
                    "url": "http://arxiv.org/abs/2501.19389v4",
                    "chunk_index": 0,
                    "source": "abstract",
                    "distance": 0.1,
                },
            ],
            "search_results": [],
        },
    ]

    for tq in test_queries:
        result = app.invoke({
            "query": tq["query"],
            "retrieved_docs": tq["retrieved_docs"],
            "search_results": tq["search_results"],
            "tool_calls": [],
            "answer": None,
            "sources": [],
            "next_action": None,
            "hop_count": 0,
            "memory_store": None,
            "memory_context": None,
            "memory_notice": None,
            "retrieval_filter": None,
            "_queried_arxiv_id": None,
        })
        print(f"Q: {tq['query']}")
        print(f"  -> {result['answer']}")
        print(f"  -> {result['sources']}")
        print(f"  -> tool_calls: {result['tool_calls']}\n")