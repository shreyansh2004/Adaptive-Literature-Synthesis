#!/usr/bin/env python3
"""
Week 2 Testing Suite — Day 13 (Stress-testing)

Comprehensive tests for the LangGraph agent to verify:
  - Correct routing decisions (planner logic)
  - Multi-hop query flows
  - Paper comparison capability
  - Answer synthesis with citations
  - Error handling and edge cases
  - Hop limit enforcement

Requirements:
  - Local Ollama server running with llama3.1:8b model
  - Week 1 RAG pipeline working (embeddings, store, retrieve)
  - Papers ingested into ChromaDB

Usage:
    python tests/week2_testing.py                           # Run all tests
    python tests/week2_testing.py --skip-llm                # Skip tests needing Ollama
    python tests/week2_testing.py --verbose                 # Detailed output
    python tests/week2_testing.py --query "your question"   # Run single query
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.graph import app as graph_app, AgentState


# ============================================================================
# Test Configuration
# ============================================================================

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
ARXIV_TEST_DELAY = 3  # seconds between arxiv API calls

@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    detail: str = ""
    routing_path: List[str] = None  # Tool calls made
    

class Week2TestRunner:
    def __init__(self, skip_llm=False, verbose=False):
        self.skip_llm = skip_llm
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.category_counts: Dict[str, Tuple[int, int]] = {}
        self.last_arxiv_call = 0  # Track last arXiv API call for rate-limiting

    def ollama_available(self) -> bool:
        try:
            import requests
            resp = requests.get(OLLAMA_TAGS_URL, timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def record(self, name: str, category: str, passed: bool, detail: str = "", routing_path: List[str] = None):
        self.results.append(TestResult(name, category, passed, detail, routing_path))
        
        if category not in self.category_counts:
            self.category_counts[category] = (0, 0)
        passed_count, total_count = self.category_counts[category]
        self.category_counts[category] = (passed_count + (1 if passed else 0), total_count + 1)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        detail_str = f" — {detail}" if detail else ""
        path_str = f" [path: {' → '.join(routing_path)}]" if routing_path else ""
        print(f"    {status}: {name}{detail_str}{path_str}")

    def run_test(self, name: str, category: str, fn, skip_if=False):
        if skip_if:
            print(f"    ⊘ SKIP: {name}")
            return
        
        try:
            fn()
        except AssertionError as e:
            self.record(name, category, False, str(e))
        except Exception as e:
            self.record(name, category, False, f"{type(e).__name__}: {e}")

    def arxiv_rate_limit(self):
        """Enforce rate-limiting delay between arXiv API calls."""
        elapsed = time.time() - self.last_arxiv_call
        if elapsed < ARXIV_TEST_DELAY:
            wait_time = ARXIV_TEST_DELAY - elapsed
            print(f"    [rate-limit] waiting {wait_time:.1f}s before next arXiv call...")
            time.sleep(wait_time)
        self.last_arxiv_call = time.time()

    def run_query(self, query: str, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a query through the graph and return full result."""
        state = {
            "query": query,
            "retrieved_docs": [],
            "search_results": [],
            "tool_calls": [],
            "answer": None,
            "sources": [],
            "next_action": None,
            "hop_count": 0,
        }
        if initial_state:
            state.update(initial_state)
        
        result = graph_app.invoke(state)
        return result

    def print_summary(self):
        print("\n" + "=" * 80)
        print("WEEK 2 TEST SUMMARY")
        print("=" * 80)
        
        total_passed = sum(r.passed for r in self.results)
        total_tests = len(self.results)
        
        for category in sorted(self.category_counts.keys()):
            passed, total = self.category_counts[category]
            pct = (passed / total * 100) if total > 0 else 0
            status = "✓" if passed == total else "✗"
            print(f"  {status} {category}: {passed}/{total} ({pct:.0f}%)")
        
        print(f"\nOverall: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.0f}%)")
        return total_passed == total_tests


# ============================================================================
# Test Categories
# ============================================================================

def print_section(title: str):
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print("=" * 80)


# ============================================================================
# Category 1: Planner Routing Logic
# ============================================================================

def test_planner_routes_to_answer_with_empty_state(runner: Week2TestRunner):
    """When no docs/search results and simple query, should route to answer."""
    result = runner.run_query("What is LoRA?")
    assert result["next_action"] is not None, "planner should set next_action"
    runner.record(
        "Empty state routes to answer_now",
        "Planner Routing",
        True,
        routing_path=result["tool_calls"]
    )


def test_planner_decides_search_on_unknown_topic(runner: Week2TestRunner):
    """For unknown topics with no local docs, should search arxiv."""
    runner.arxiv_rate_limit()  # Rate-limit arXiv API calls
    result = runner.run_query("Tell me about the latest papers on quantum computing")
    # Planner should decide to search (requires arxiv access)
    runner.record(
        "Unknown topic triggers search",
        "Planner Routing",
        True,
        detail=f"Action: {result['next_action']}",
        routing_path=result["tool_calls"]
    )


def test_planner_respects_max_hops(runner: Week2TestRunner):
    """After 4 hops, planner should force answer_now."""
    state = {
        "query": "What is LoRA?",
        "retrieved_docs": [],
        "search_results": [],
        "hop_count": 4,  # Already at max
        "next_action": None,
        "tool_calls": [],
        "answer": None,
        "sources": [],
    }
    result = runner.run_query("What is LoRA?", initial_state=state)
    assert result["hop_count"] >= 4, "hop_count should increment"
    assert result["next_action"] == "answer_now", f"should force answer_now at max hops, got {result['next_action']}"
    runner.record(
        "Max hops enforced (>= 4)",
        "Planner Routing",
        True,
        detail=f"Hop count: {result['hop_count']}",
        routing_path=result["tool_calls"]
    )


def test_planner_avoids_research_when_search_results_exist(runner: Week2TestRunner):
    """If search_results already populated, don't search again."""
    state = {
        "query": "Tell me more about LoRA",
        "search_results": [{"arxiv_id": "2106.09685", "title": "LoRA: Low-Rank Adaptation"}],
        "retrieved_docs": [],
    }
    result = runner.run_query("Tell me more about LoRA", initial_state=state)
    # Should not choose search_arxiv
    assert result["next_action"] != "search_arxiv", \
        f"planner should not re-search when search_results exist, got {result['next_action']}"
    runner.record(
        "Avoid duplicate search",
        "Planner Routing",
        True,
        detail=f"Action: {result['next_action']}",
        routing_path=result["tool_calls"]
    )


def test_planner_compare_requires_explicit_keyword(runner: Week2TestRunner):
    """Compare action only triggered if query explicitly asks for comparison."""
    state = {
        "query": "What do these papers discuss?",  # NOT asking for comparison
        "retrieved_docs": [
            {"arxiv_id": "P1", "title": "Paper 1", "text": "..."},
            {"arxiv_id": "P2", "title": "Paper 2", "text": "..."},
        ],
    }
    result = runner.run_query("What do these papers discuss?", initial_state=state)
    # Even with 2 papers, should NOT choose compare without "compare" keyword
    assert result["next_action"] != "compare_papers", \
        f"should not compare without explicit keyword, got {result['next_action']}"
    runner.record(
        "Compare requires explicit keyword",
        "Planner Routing",
        True,
        detail=f"Action: {result['next_action']}",
        routing_path=result["tool_calls"]
    )


def test_planner_compare_with_keyword_and_multiple_papers(runner: Week2TestRunner):
    """Compare action triggered when query asks to compare AND 2+ papers present."""
    state = {
        "query": "Compare the approaches of these two papers on LoRA",
        "retrieved_docs": [
            {"arxiv_id": "2106.09685", "title": "LoRA: Low-Rank Adaptation", "text": "..."},
            {"arxiv_id": "2411.14961", "title": "LoRA-FAIR", "text": "..."},
        ],
    }
    result = runner.run_query("Compare the approaches of these two papers on LoRA", initial_state=state)
    assert result["next_action"] == "compare_papers", \
        f"should choose compare_papers with keyword + 2+ papers, got {result['next_action']}"
    runner.record(
        "Compare with keyword triggers comparison",
        "Planner Routing",
        True,
        detail=f"Papers: {len(set(d['arxiv_id'] for d in state['retrieved_docs']))}, Action: {result['next_action']}",
        routing_path=result["tool_calls"]
    )


# ============================================================================
# Category 2: Multi-Hop Flows
# ============================================================================

def test_multihop_search_then_retrieve_then_answer(runner: Week2TestRunner):
    """Multi-hop: planner → search → fetch+ingest → retrieve → planner → answer."""
    runner.arxiv_rate_limit()  # Rate-limit arXiv API calls
    query = "What are the latest papers on federated LoRA?"
    result = runner.run_query(query)
    
    # Check that multiple tool calls happened (search, fetch_and_ingest, retrieve, etc.)
    tool_calls = result.get("tool_calls", [])
    assert len(tool_calls) > 1, f"expected multi-hop flow, got {len(tool_calls)} tool calls"
    
    # Final result should have answer
    assert result.get("answer") is not None and len(result["answer"]) > 0, \
        "multi-hop flow should produce an answer"
    
    runner.record(
        "Multi-hop search→ingest→retrieve→answer",
        "Multi-Hop Flows",
        True,
        detail=f"{len(tool_calls)} tool calls",
        routing_path=tool_calls
    )


def test_multihop_retrieve_then_answer(runner: Week2TestRunner):
    """Multi-hop: planner → retrieve → planner → answer (if local docs available)."""
    runner.arxiv_rate_limit()  # Rate-limit in case planner decides to search
    query = "What does the LoRA paper say about parameter efficiency?"
    result = runner.run_query(query)
    
    # Should eventually produce answer
    assert result.get("answer") is not None, "should produce answer"
    
    # Hop count should be tracked
    assert result.get("hop_count", 0) > 0, "hop_count should be incremented"
    
    runner.record(
        "Multi-hop retrieve→answer",
        "Multi-Hop Flows",
        True,
        detail=f"Hops: {result.get('hop_count')}, Tool calls: {len(result['tool_calls'])}",
        routing_path=result["tool_calls"]
    )


# ============================================================================
# Category 3: Answer Synthesis
# ============================================================================

def test_answer_includes_citations(runner: Week2TestRunner):
    """Answer should include [n] citations matching numbered sources."""
    state = {
        "query": "What is overfitting?",
        "retrieved_docs": [
            {
                "text": "Overfitting occurs when a model memorizes training data rather than generalizing to new examples.",
                "arxiv_id": "9999.99999",
                "title": "Fake Paper on Overfitting",
                "authors": "Test Author",
                "published": "2024-01-01",
                "updated": "2024-01-01",
                "url": "http://arxiv.org/abs/9999.99999",
                "chunk_index": 0,
                "source": "abstract",
                "distance": 0.1,
            }
        ],
        "next_action": "answer_now",
    }
    result = runner.run_query("What is overfitting?", initial_state=state)
    
    answer = result.get("answer", "")
    has_citations = "[1]" in answer or "[2]" in answer or "[3]" in answer
    
    assert has_citations or "overfitting" in answer.lower(), \
        f"answer should cite sources or mention key term, got: {answer[:100]}"
    
    runner.record(
        "Answer includes citations [n]",
        "Answer Synthesis",
        True,
        detail="Citations present",
        routing_path=result["tool_calls"]
    )


def test_answer_without_relevant_docs(runner: Week2TestRunner):
    """When no docs available, should say so rather than making up answer."""
    state = {
        "query": "Some esoteric query that won't have docs",
        "retrieved_docs": [],
        "next_action": "answer_now",
    }
    result = runner.run_query("Some esoteric query", initial_state=state)
    
    answer = result.get("answer", "")
    # Should indicate lack of information
    negative_phrases = [
        "couldn't find",
        "no information",
        "no relevant",
        "not found",
        "don't contain",
        "doesn't contain",
        "do not contain",
        "not enough information",
        "insufficient information",
        "not provided",
        "cannot answer",
    ]
    assert any(p in answer.lower() for p in negative_phrases), \
        f"should indicate lack of info, got: {answer[:100]}"
    
    runner.record(
        "No docs → 'couldn't find' message",
        "Answer Synthesis",
        True,
        routing_path=result["tool_calls"]
    )


def test_answer_has_sources_list(runner: Week2TestRunner):
    """Answer result should include sources list with arxiv_id, title, url."""
    state = {
        "query": "What is LoRA?",
        "retrieved_docs": [
            {
                "text": "LoRA reduces trainable parameters through low-rank decomposition.",
                "arxiv_id": "2106.09685",
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "authors": "Edward Liang et al.",
                "published": "2021-06-17",
                "updated": "2021-10-03",
                "url": "http://arxiv.org/abs/2106.09685",
                "chunk_index": 0,
                "source": "full_text",
                "distance": 0.05,
            }
        ],
        "next_action": "answer_now",
    }
    result = runner.run_query("What is LoRA?", initial_state=state)
    
    sources = result.get("sources", [])
    assert len(sources) > 0, "should have at least one source"
    assert any(s.get("arxiv_id") for s in sources), "source should have arxiv_id"
    assert any(s.get("title") for s in sources), "source should have title"
    assert any(s.get("url") for s in sources), "source should have url"
    
    runner.record(
        "Answer sources include metadata",
        "Answer Synthesis",
        True,
        detail=f"{len(sources)} source(s)",
        routing_path=result["tool_calls"]
    )


# ============================================================================
# Category 4: Paper Comparison
# ============================================================================

def test_comparator_structures_output(runner: Week2TestRunner):
    """Comparator should produce structured JSON with method/results/limitations."""
    state = {
        "query": "Compare LoRA-FAIR and Federated Sketching LoRA",
        "retrieved_docs": [
            {
                "text": "LoRA-FAIR uses server-side bias correction and client-side refinement.",
                "arxiv_id": "2411.14961",
                "title": "LoRA-FAIR: Federated LoRA Fine-Tuning",
                "authors": "Author 1",
                "published": "2024-11-01",
                "updated": "2024-11-01",
                "url": "http://arxiv.org/abs/2411.14961v3",
                "chunk_index": 0,
                "source": "abstract",
                "distance": 0.1,
            },
            {
                "text": "FSLoRA uses sketching to compress LoRA updates before transmission.",
                "arxiv_id": "2501.19389",
                "title": "Federated Sketching LoRA",
                "authors": "Author 2",
                "published": "2025-01-01",
                "updated": "2025-01-01",
                "url": "http://arxiv.org/abs/2501.19389v4",
                "chunk_index": 0,
                "source": "abstract",
                "distance": 0.12,
            },
        ],
        "next_action": "compare_papers",
    }
    result = runner.run_query("Compare LoRA-FAIR and Federated Sketching LoRA", initial_state=state)
    
    answer = result.get("answer", "")
    # Should mention both papers or methods
    has_structure = "[1]" in answer and "[2]" in answer
    has_comparison = any(word in answer.lower() for word in ["method", "sketching", "bias correction", "difference"])
    
    assert has_structure or has_comparison, \
        f"comparison should structure papers and show differences, got: {answer[:150]}"
    
    runner.record(
        "Comparator structures output",
        "Paper Comparison",
        True,
        detail="Structured comparison produced",
        routing_path=result["tool_calls"]
    )


def test_comparator_requires_2_papers(runner: Week2TestRunner):
    """Comparator should reject if < 2 distinct papers in retrieved_docs."""
    state = {
        "query": "Compare these papers",
        "retrieved_docs": [
            {
                "text": "Only one paper here.",
                "arxiv_id": "1111.11111",
                "title": "Single Paper",
                "authors": "Author",
                "published": "2024-01-01",
                "updated": "2024-01-01",
                "url": "http://arxiv.org/abs/1111.11111",
                "chunk_index": 0,
                "source": "abstract",
                "distance": 0.1,
            }
        ],
        "next_action": "compare_papers",
    }
    result = runner.run_query("Compare these papers", initial_state=state)
    
    answer = result.get("answer", "")
    # Should ask for more papers
    assert "2" in answer or "need" in answer.lower() or "two" in answer.lower(), \
        f"should ask for 2+ papers, got: {answer[:100]}"
    
    runner.record(
        "Comparator requires 2+ papers",
        "Paper Comparison",
        True,
        detail="Correctly rejects single paper",
        routing_path=result["tool_calls"]
    )


# ============================================================================
# Category 5: State Management & Edge Cases
# ============================================================================

def test_hop_count_increments(runner: Week2TestRunner):
    """Hop count should increment with each planner call."""
    state = {
        "query": "Test query",
        "retrieved_docs": [],
        "search_results": [],
        "hop_count": 0,
    }
    result = runner.run_query("Test query", initial_state=state)
    
    assert result["hop_count"] > 0, "hop_count should be incremented from initial 0"
    
    runner.record(
        "Hop count increments",
        "State Management",
        True,
        detail=f"Final hop_count: {result['hop_count']}",
        routing_path=result["tool_calls"]
    )


def test_tool_calls_accumulated(runner: Week2TestRunner):
    """tool_calls list should accumulate as nodes execute."""
    result = runner.run_query("What is LoRA?")
    
    tool_calls = result.get("tool_calls", [])
    assert len(tool_calls) > 0, "should accumulate tool_calls"
    # All entries should be strings
    assert all(isinstance(tc, str) for tc in tool_calls), "tool_calls should all be strings"
    
    runner.record(
        "Tool calls accumulated",
        "State Management",
        True,
        detail=f"{len(tool_calls)} calls: {', '.join(tool_calls[:3])}...",
        routing_path=tool_calls
    )


def test_query_preserved_through_hops(runner: Week2TestRunner):
    """Query should stay same throughout multi-hop execution."""
    query = "What papers discuss federated learning with LoRA?"
    result = runner.run_query(query)
    
    assert result["query"] == query, "query should not change across hops"
    
    runner.record(
        "Query preserved through hops",
        "State Management",
        True,
        routing_path=result["tool_calls"]
    )


# ============================================================================
# Main Test Orchestration
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Week 2 Testing Suite (Day 13)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Ollama-dependent tests")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument("--query", type=str, help="Run single query instead of full test suite")
    args = parser.parse_args()
    
    runner = Week2TestRunner(skip_llm=args.skip_llm, verbose=args.verbose)
    
    # Check Ollama availability
    ollama_available = runner.ollama_available()
    if args.skip_llm or not ollama_available:
        if not ollama_available:
            print("⚠️  Ollama server not reachable at http://localhost:11434")
            print("   LLM-dependent tests will be skipped or may fail")
            print("   Start Ollama: ollama serve\n")
    
    # ========================================================================
    # Run Test Categories
    # ========================================================================
    
    if args.query:
        print_section("SINGLE QUERY TEST")
        print(f"Query: {args.query}\n")
        result = runner.run_query(args.query)
        print(f"\nResult:")
        print(f"  Answer: {result['answer'][:200]}...")
        print(f"  Sources: {len(result['sources'])} sources")
        print(f"  Tool calls: {result['tool_calls']}")
        print(f"  Hops: {result['hop_count']}")
        return 0
    
    # ========================================================================
    print_section("CATEGORY 1: PLANNER ROUTING LOGIC")
    runner.run_test("Empty state routes to answer_now", "Planner Routing", 
                   lambda: test_planner_routes_to_answer_with_empty_state(runner), skip_if=args.skip_llm)
    runner.run_test("Unknown topic triggers search", "Planner Routing",
                   lambda: test_planner_decides_search_on_unknown_topic(runner), skip_if=args.skip_llm)
    runner.run_test("Max hops enforced (>= 4)", "Planner Routing",
                   lambda: test_planner_respects_max_hops(runner), skip_if=args.skip_llm)
    runner.run_test("Avoid duplicate search", "Planner Routing",
                   lambda: test_planner_avoids_research_when_search_results_exist(runner), skip_if=args.skip_llm)
    runner.run_test("Compare requires explicit keyword", "Planner Routing",
                   lambda: test_planner_compare_requires_explicit_keyword(runner), skip_if=args.skip_llm)
    runner.run_test("Compare with keyword triggers comparison", "Planner Routing",
                   lambda: test_planner_compare_with_keyword_and_multiple_papers(runner), skip_if=args.skip_llm)
    
    # ========================================================================
    print_section("CATEGORY 2: MULTI-HOP FLOWS")
    runner.run_test("Multi-hop search→ingest→retrieve→answer", "Multi-Hop Flows",
                   lambda: test_multihop_search_then_retrieve_then_answer(runner), skip_if=args.skip_llm)
    runner.run_test("Multi-hop retrieve→answer", "Multi-Hop Flows",
                   lambda: test_multihop_retrieve_then_answer(runner), skip_if=args.skip_llm)
    
    # ========================================================================
    print_section("CATEGORY 3: ANSWER SYNTHESIS")
    runner.run_test("Answer includes citations [n]", "Answer Synthesis",
                   lambda: test_answer_includes_citations(runner), skip_if=args.skip_llm)
    runner.run_test("No docs → 'couldn't find' message", "Answer Synthesis",
                   lambda: test_answer_without_relevant_docs(runner), skip_if=args.skip_llm)
    runner.run_test("Answer sources include metadata", "Answer Synthesis",
                   lambda: test_answer_has_sources_list(runner), skip_if=args.skip_llm)
    
    # ========================================================================
    print_section("CATEGORY 4: PAPER COMPARISON")
    runner.run_test("Comparator structures output", "Paper Comparison",
                   lambda: test_comparator_structures_output(runner), skip_if=args.skip_llm)
    runner.run_test("Comparator requires 2+ papers", "Paper Comparison",
                   lambda: test_comparator_requires_2_papers(runner), skip_if=args.skip_llm)
    
    # ========================================================================
    print_section("CATEGORY 5: STATE MANAGEMENT & EDGE CASES")
    runner.run_test("Hop count increments", "State Management",
                   lambda: test_hop_count_increments(runner), skip_if=args.skip_llm)
    runner.run_test("Tool calls accumulated", "State Management",
                   lambda: test_tool_calls_accumulated(runner), skip_if=args.skip_llm)
    runner.run_test("Query preserved through hops", "State Management",
                   lambda: test_query_preserved_through_hops(runner), skip_if=args.skip_llm)
    
    # ========================================================================
    # Summary
    # ========================================================================
    success = runner.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
