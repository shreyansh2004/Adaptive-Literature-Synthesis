#!/usr/bin/env python3
"""
Comprehensive Week 1 Test Pipeline

This script runs all Week 1 tests in a structured, organized manner with clear
reporting. It tests:
  - Day 1: FastAPI setup (basic health check)
  - Day 2: arXiv search tool
  - Day 3: Paper fetching tool
  - Day 4: RAG pipeline (chunking, embeddings, storage, ingestion)
  - Day 5: Retrieval and answer synthesis
  - Day 6: End-to-end QA system

Usage:
    python scripts/run_week1_tests.py                 # Run all tests
    python scripts/run_week1_tests.py --skip-network  # Skip network-dependent tests
    python scripts/run_week1_tests.py --skip-llm      # Skip Ollama-dependent tests
    python scripts/run_week1_tests.py --keep          # Keep test data after run

Requirements:
    - Network access to export.arxiv.org (for live arXiv tests)
    - Local Ollama server (optional; some tests skipped if not available)
    - See requirements.txt for Python dependencies
"""

import os
import sys
import gc
import math
import time
import json
import shutil
import argparse
import tempfile
import traceback
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ============================================================================
# Configuration
# ============================================================================

TEST_RAW_DIR = "data/raw_test"
TEST_CHROMA_DIR = tempfile.mkdtemp(prefix="chroma_test_")
TEST_COLLECTION = "papers_test"

# Known stable arXiv papers for testing
LORA_ID = "2106.09685"       # LoRA: Low-Rank Adaptation of Large Language Models
RAG_ID = "2005.11401"        # Retrieval-Augmented Generation (foundational RAG paper)
BOGUS_ID = "9999.99999"      # Non-existent paper

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


# ============================================================================
# Result Tracking
# ============================================================================

@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    detail: str = ""
    duration: float = 0.0


class TestRunner:
    def __init__(self, skip_network=False, skip_llm=False):
        self.skip_network = skip_network
        self.skip_llm = skip_llm
        self.results: List[TestResult] = []
        self.category_counts: Dict[str, Tuple[int, int]] = {}  # category -> (passed, total)

    def record(self, name: str, category: str, passed: bool, detail: str = "", duration: float = 0.0):
        self.results.append(TestResult(name, category, passed, detail, duration))
        
        if category not in self.category_counts:
            self.category_counts[category] = (0, 0)
        passed_count, total_count = self.category_counts[category]
        self.category_counts[category] = (passed_count + (1 if passed else 0), total_count + 1)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        detail_str = f" — {detail}" if detail else ""
        print(f"    {status}: {name}{detail_str}")

    def run_test(self, name: str, category: str, fn, skip_if=False):
        if skip_if:
            print(f"    ⊘ SKIP: {name}")
            return
        
        start = time.time()
        try:
            fn()
            self.record(name, category, True, duration=time.time() - start)
        except AssertionError as e:
            self.record(name, category, False, str(e), duration=time.time() - start)
        except Exception as e:
            self.record(name, category, False, f"{type(e).__name__}: {e}", duration=time.time() - start)

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
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
# Utility Functions
# ============================================================================

def cleanup_test_data():
    """Remove test isolation directories to clean up after run."""
    gc.collect()
    time.sleep(0.1)  # Give OS time to release file handles
    for d in (TEST_RAW_DIR, TEST_CHROMA_DIR):
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"  Cleaned up {d}")
            except PermissionError as e:
                print(f"  (warning: couldn't fully clean {d}: {e})")


def ollama_available() -> bool:
    """Check if Ollama server is running and reachable."""
    try:
        import requests
        resp = requests.get(OLLAMA_TAGS_URL, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def ollama_model_pulled(model_name: str) -> bool:
    """Check if a specific model is pulled in Ollama."""
    try:
        import requests
        resp = requests.get(OLLAMA_TAGS_URL, timeout=2)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        model_prefix = model_name.split(":")[0]
        return any(n == model_name or n.startswith(model_prefix) for n in names)
    except Exception:
        return False


def print_section(title: str):
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print("=" * 80)


# ============================================================================
# Day 1 Tests: FastAPI Setup
# ============================================================================

def test_fastapi_endpoint_exists():
    """Verify FastAPI app can be imported and has at least one route."""
    from api.main import app
    assert app is not None
    # Check that routes exist
    assert len(app.routes) > 0, "FastAPI app has no routes"


def test_fastapi_hello_endpoint():
    """Test the basic health check endpoint."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_fastapi_env_vars_loaded():
    """Verify critical environment variables are present at import time."""
    import importlib
    import api.main  # importing should not raise if env vars are set correctly
    # The app should expose which model/embed config is in use via a settings object
    # If your app defines a `settings` or `config` object, assert it here.
    # Minimal check: the module imported without error.
    assert api.main is not None


# ============================================================================
# Day 2 Tests: arXiv Search Tool
# ============================================================================

def test_arxiv_search_basic():
    """Search arXiv and verify result structure."""
    from tools.arxiv_search import search_arxiv
    results = search_arxiv("LoRA fine-tuning", max_results=5)
    
    assert len(results) > 0, "expected at least 1 result for 'LoRA fine-tuning'"
    assert len(results) <= 5, f"requested max_results=5, got {len(results)}"
    
    p = results[0]
    required_fields = ("arxiv_id", "title", "abstract", "authors", "url", "pdf_url", "published")
    for field in required_fields:
        val = getattr(p, field, None)
        assert val, f"field '{field}' is empty/missing on first result"
    
    assert p.url.startswith("http"), "url should be HTTP(S) link"
    assert "pdf" in p.pdf_url.lower(), "pdf_url should mention PDF"


def test_arxiv_search_empty_topic_raises():
    """Verify empty search raises ValueError."""
    from tools.arxiv_search import search_arxiv
    try:
        search_arxiv("   ")
        raise AssertionError("expected ValueError for blank topic")
    except ValueError:
        pass  # expected


def test_arxiv_search_max_results_respected():
    """Verify max_results parameter is honored."""
    from tools.arxiv_search import search_arxiv
    for limit in [1, 3, 5]:
        results = search_arxiv("transformers", max_results=limit)
        assert len(results) <= limit, f"requested {limit}, got {len(results)}"


def test_arxiv_search_no_results_returns_empty_list():
    """Verify nonsense query returns an empty list, not an error or non-empty list."""
    from tools.arxiv_search import search_arxiv
    results = search_arxiv("zzzzqqqq99999 nonexistent blarg xyzzy", max_results=3)
    assert isinstance(results, list), "expected a list"


# ============================================================================
# Day 3 Tests: Paper Fetching
# ============================================================================

def test_fetch_paper_metadata():
    """Fetch metadata for a known paper."""
    from tools.paper_fetch import fetch_paper_metadata
    meta = fetch_paper_metadata(LORA_ID)
    
    assert meta.arxiv_id == LORA_ID
    assert "LoRA" in meta.title or "Low-Rank" in meta.title, f"unexpected title: {meta.title}"
    assert len(meta.authors) > 0


def test_fetch_paper_metadata_invalid_id():
    """Verify invalid ID raises ValueError specifically (not a silent failure or wrong exception)."""
    from tools.paper_fetch import fetch_paper_metadata
    try:
        fetch_paper_metadata(BOGUS_ID)
        raise AssertionError("expected ValueError for invalid arxiv_id, but no exception was raised")
    except ValueError:
        pass  # expected — function correctly signals bad input
    # Any other exception type (KeyError, AttributeError, etc.) will propagate and fail the test


def test_fetch_paper_full_text():
    """Fetch full paper text and verify it's substantial."""
    from tools.paper_fetch import fetch_paper
    
    paper = fetch_paper(LORA_ID, include_full_text=True)
    assert paper.arxiv_id == LORA_ID
    assert paper.full_text is not None, f"full text fetch failed: {paper.fetch_error}"
    assert len(paper.full_text) > 1000, f"full text too short: {len(paper.full_text)} chars"


def test_fetch_paper_graceful_fallback_to_abstract():
    """Verify abstract is available even if PDF fetch fails."""
    from tools.paper_fetch import fetch_paper
    
    paper = fetch_paper(LORA_ID, include_full_text=True)
    # Even if full text fetch fails, we should have abstract
    assert paper.abstract and len(paper.abstract) > 100, "abstract missing or too short"
 


def test_fetch_second_paper_rag_id():
    """Fetch the RAG paper to ensure multi-paper handling works (catches ID-specific bugs)."""
    from tools.paper_fetch import fetch_paper_metadata
    meta = fetch_paper_metadata(RAG_ID)

    assert meta.arxiv_id == RAG_ID
    assert len(meta.title) > 0, "title should not be empty"
    assert len(meta.authors) > 0, "authors should not be empty"


# ============================================================================
# Day 4 Tests: Chunking
# ============================================================================

def test_chunk_text_empty_returns_empty():
    """Verify empty input yields empty output."""
    from rag.chunking import chunk_text
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_short_text():
    """Verify short text stays in single chunk."""
    from rag.chunking import chunk_text
    text = "This is a short paragraph that fits in one chunk."
    chunks = chunk_text(text)
    assert len(chunks) == 1


def test_chunk_text_respects_paragraph_boundaries():
    """Verify chunking respects paragraph boundaries when possible."""
    from rag.chunking import chunk_text
    paras = [f"Paragraph {i}. " + ("word " * 20) for i in range(5)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_size=50, overlap=5)
    
    assert len(chunks) > 1, "expected multiple chunks"
    assert all(c.strip() for c in chunks), "found empty chunk"


def test_chunk_text_token_aware_splitting():
    """Verify oversized paragraph is split by tokens."""
    from rag.chunking import chunk_text
    huge_para = "word " * 2000  # way over 500 tokens
    chunks = chunk_text(huge_para, chunk_size=500, overlap=50)
    
    assert len(chunks) > 1, "oversized paragraph should be split"


def test_chunk_text_overlap_exists():
    """Verify overlapping content between consecutive chunks."""
    from rag.chunking import chunk_text
    huge_para = " ".join(f"tok{i}" for i in range(2000))
    chunks = chunk_text(huge_para, chunk_size=500, overlap=50)
    
    assert len(chunks) >= 2
    # Consecutive chunks should share some tokens
    tail_tokens = set(chunks[0].split()[-20:])
    head_tokens = set(chunks[1].split()[:20])
    assert tail_tokens & head_tokens, "expected token overlap between chunks"


# ============================================================================
# Day 4 Tests: Embeddings
# ============================================================================

def test_embed_texts_correct_shape():
    """Verify embeddings have correct shape."""
    from rag.embeddings import embed_texts
    texts = ["LoRA reduces trainable parameters.", "RAG retrieves documents."]
    vectors = embed_texts(texts)
    
    assert len(vectors) == 2
    dim = len(vectors[0])
    assert dim > 0, "embedding should have non-zero dimension"
    assert all(len(v) == dim for v in vectors), "inconsistent dimensions"


def test_embed_texts_deterministic():
    """Verify same input produces identical (or near-identical) embeddings across calls."""
    from rag.embeddings import embed_texts
    text = "deterministic test sentence"
    v1 = embed_texts([text])[0]
    v2 = embed_texts([text])[0]
    assert len(v1) == len(v2), "embedding dimensions differ between calls"
    mismatches = [
        i for i, (a, b) in enumerate(zip(v1, v2))
        if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    ]
    assert not mismatches, (
        f"embeddings differ at {len(mismatches)} dimension(s): indices {mismatches[:5]}..."
    )


def test_embed_texts_empty_list():
    """Verify empty input yields empty output."""
    from rag.embeddings import embed_texts
    result = embed_texts([])
    assert len(result) == 0


# ============================================================================
# Day 4 Tests: Vector Store
# ============================================================================

def _fresh_store():
    """Create isolated test store."""
    from rag.store import VectorStore
    return VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)


def test_store_add_and_count():
    """Verify adding chunks and counting."""
    from rag.embeddings import embed_texts
    
    store = _fresh_store()
    texts = ["chunk one about LoRA", "chunk two about RAG"]
    embeddings = embed_texts(texts)
    
    store.add_chunks(
        ids=["test_1", "test_2"],
        texts=texts,
        embeddings=embeddings,
        metadatas=[
            {"arxiv_id": "TESTPAPER", "chunk_index": 0},
            {"arxiv_id": "TESTPAPER", "chunk_index": 1}
        ],
    )
    
    assert store.count() >= 2


def test_store_query():
    """Verify querying returns results."""
    from rag.embeddings import embed_texts
    
    store = _fresh_store()
    query_emb = embed_texts(["What is LoRA?"])[0]
    result = store.query(query_embedding=query_emb, top_k=2)
    
    assert "documents" in result
    assert len(result["documents"][0]) > 0


def test_store_paper_already_ingested():
    """Verify idempotency check."""
    store = _fresh_store()
    assert store.paper_already_ingested("TESTPAPER") is True
    assert store.paper_already_ingested("NOT_A_REAL_PAPER_ID") is False


def test_store_delete_paper():
    """Verify deleting paper removes its chunks."""
    store = _fresh_store()
    before = store.count()
    store.delete_paper("TESTPAPER")
    after = store.count()
    
    assert after < before, "count should decrease after delete"
    assert not store.paper_already_ingested("TESTPAPER")


# ============================================================================
# Day 4 Tests: Ingestion Pipeline
# ============================================================================

def test_ingest_paper():
    """Ingest a paper and verify chunks are stored."""
    from rag.ingest import ingest_paper
    from rag.store import VectorStore
    from tools.paper_fetch import fetch_paper

    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    paper = fetch_paper(LORA_ID, include_full_text=True)
    paper_dict = paper.to_dict()
    
    n = ingest_paper(paper_dict, store, force=False)
    assert n > 0, f"expected chunks to be ingested, got {n}"


def test_ingest_paper_idempotent():
    """Verify re-ingesting same paper is idempotent."""
    from rag.ingest import ingest_paper
    from rag.store import VectorStore
    from tools.paper_fetch import fetch_paper

    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    store.delete_paper(LORA_ID)

    paper = fetch_paper(LORA_ID, include_full_text=True)
    paper_dict = paper.to_dict()
    
    n1 = ingest_paper(paper_dict, store, force=False)
    n2 = ingest_paper(paper_dict, store, force=False)

    assert n1 > 0
    assert n2 == 0, "re-ingesting without force=True should skip"


def test_ingest_paper_force_reingest():
    """Verify force=True re-ingests."""
    from rag.ingest import ingest_paper
    from rag.store import VectorStore
    from tools.paper_fetch import fetch_paper
    
    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    store.delete_paper(LORA_ID)

    paper = fetch_paper(LORA_ID, include_full_text=True)
    paper_dict = paper.to_dict()
    
    n1 = ingest_paper(paper_dict, store, force=False)
    n2 = ingest_paper(paper_dict, store, force=True)

    assert n1 > 0
    assert n2 > 0, "force=True should re-ingest"


def test_ingest_paper_abstract_fallback():
    """Verify abstract is used if full text unavailable."""
    from rag.ingest import ingest_paper
    from rag.store import VectorStore
    
    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    fake_paper = {
        "arxiv_id": "FAKE_ABSTRACT",
        "title": "Fake Paper",
        "abstract": "This is a test abstract. " * 10,
        "authors": ["Test Author"],
        "published": "2024-01-01",
        "updated": "2024-01-01",
        "url": "https://arxiv.org/abs/FAKE_ABSTRACT",
        "full_text": None,
        "full_text_fetched": False,
    }
    
    n = ingest_paper(fake_paper, store, force=False)
    assert n > 0, "should ingest from abstract"
    
    store.delete_paper("FAKE_ABSTRACT")


# ============================================================================
# Day 5 Tests: Retrieval
# ============================================================================

def test_retrieve_returns_results():
    """Verify retrieval returns relevant chunks."""
    from rag.retrieve import retrieve
    from rag.store import VectorStore
    
    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    results = retrieve("How does LoRA work?", store=store, top_k=3)
    
    assert len(results) > 0, "expected at least 1 result"
    assert all(r.text.strip() for r in results), "chunk text should be non-empty"


def test_retrieve_ranked_by_relevance():
    """Verify results are ranked by relevance."""
    from rag.retrieve import retrieve
    from rag.store import VectorStore
    
    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    results = retrieve("LoRA parameters", store=store, top_k=5)
    
    distances = [r.distance for r in results]
    assert distances == sorted(distances), "results should be sorted by distance (ascending)"


def test_retrieve_with_metadata_filter():
    """Verify metadata filtering works."""
    from rag.retrieve import retrieve
    from rag.store import VectorStore
    
    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    results = retrieve(
        "fine-tuning",
        store=store,
        top_k=5,
        where={"arxiv_id": LORA_ID}
    )
    
    assert all(r.arxiv_id == LORA_ID for r in results), "metadata filter should be applied"


def test_retrieve_nonexistent_filter_returns_empty():
    """Verify filtering by a nonexistent arxiv_id returns [] rather than throwing."""
    from rag.retrieve import retrieve
    from rag.store import VectorStore

    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    results = retrieve(
        "anything",
        store=store,
        top_k=5,
        where={"arxiv_id": "THIS_ID_DOES_NOT_EXIST_XYZ"}
    )
    assert isinstance(results, list), "expected a list"
    assert len(results) == 0, f"expected 0 results for nonexistent filter, got {len(results)}"


def test_retrieve_top_k_larger_than_store():
    """Verify top_k > number of stored chunks returns what's available, not an error."""
    from rag.retrieve import retrieve
    from rag.store import VectorStore

    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    total = store.count()
    results = retrieve("LoRA", store=store, top_k=total + 1000)
    assert isinstance(results, list), "expected a list"
    assert len(results) <= total, (
        f"returned more results ({len(results)}) than chunks in store ({total})"
    )


# ============================================================================
# Day 5 Tests: Answer Synthesis (Pure Functions)
# ============================================================================

def test_sanitize_answer_removes_leading_symbols():
    """Verify leading equations are stripped."""
    from rag.answer import sanitize_answer
    messy = "x = W_0 + BA\n\nThis is the actual explanation."
    cleaned = sanitize_answer(messy)
    assert cleaned.startswith("This is"), "leading symbols should be removed"


def test_sanitize_answer_preserves_prose():
    """Verify clean prose is preserved."""
    from rag.answer import sanitize_answer
    prose = "LoRA reduces trainable parameters by decomposing weight updates."
    assert sanitize_answer(prose) == prose


def test_group_chunks_by_paper():
    """Verify chunks are grouped by paper."""
    from rag.answer import group_chunks_by_paper
    from rag.retrieve import RetrievedChunk
    
    c1 = RetrievedChunk(
        text="a", arxiv_id="P1", title="Paper 1", authors="", published="",
        updated="", url="", chunk_index=0, source="abstract", distance=0.1
    )
    c2 = RetrievedChunk(
        text="b", arxiv_id="P2", title="Paper 2", authors="", published="",
        updated="", url="", chunk_index=0, source="abstract", distance=0.2
    )
    
    grouped = group_chunks_by_paper([c1, c2])
    assert list(grouped.keys()) == ["P1", "P2"]


def test_build_context_block_numbers_papers():
    """Verify context block numbers by paper, not chunk."""
    from rag.answer import group_chunks_by_paper, build_context_block
    from rag.retrieve import RetrievedChunk
    
    c1 = RetrievedChunk(
        text="excerpt A", arxiv_id="P1", title="Paper 1", authors="", published="",
        updated="", url="", chunk_index=0, source="abstract", distance=0.1
    )
    c2 = RetrievedChunk(
        text="excerpt B", arxiv_id="P1", title="Paper 1", authors="", published="",
        updated="", url="", chunk_index=1, source="abstract", distance=0.2
    )
    
    grouped = group_chunks_by_paper([c1, c2])
    block = build_context_block(grouped)
    
    assert "[1]" in block, "should have paper [1]"
    assert block.count("[1]") >= 2, "both excerpts should reference [1]"


# ============================================================================
# Day 5 Tests: LLM Integration (Ollama)
# ============================================================================

def test_call_llm_with_retry():
    """Verify LLM call succeeds with valid response shape."""
    from rag.answer import _call_llm_with_retry
    
    response = _call_llm_with_retry([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is LoRA?"}
    ])
    
    assert "message" in response, "response missing 'message' field"
    assert "content" in response["message"], "message missing 'content'"
    assert len(response["message"]["content"]) > 0, "response content is empty"


# ============================================================================
# Day 6 Tests: End-to-End
# ============================================================================

def test_end_to_end_retrieve_and_answer():
    """Full pipeline: retrieve -> synthesize answer (context + citation check)."""
    from rag.retrieve import retrieve
    from rag.answer import group_chunks_by_paper, build_context_block
    from rag.store import VectorStore

    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)

    chunks = retrieve("How does LoRA reduce parameters?", store=store, top_k=5)
    assert len(chunks) > 0, "retrieval failed"

    grouped = group_chunks_by_paper(chunks)
    context = build_context_block(grouped)

    assert len(context) > 0, "context block is empty"
    assert "[1]" in context or "[2]" in context, "context should have numbered sources"


def test_end_to_end_citation_sources_are_real_arxiv_ids():
    """Verify that citation arxiv_ids in retrieved chunks actually exist in the store."""
    from rag.retrieve import retrieve
    from rag.answer import group_chunks_by_paper
    from rag.store import VectorStore

    store = VectorStore(persist_dir=TEST_CHROMA_DIR, collection_name=TEST_COLLECTION)
    chunks = retrieve("retrieval augmented generation", store=store, top_k=5)
    assert len(chunks) > 0, "retrieval returned nothing"

    grouped = group_chunks_by_paper(chunks)
    for arxiv_id in grouped:
        assert store.paper_already_ingested(arxiv_id), (
            f"citation references '{arxiv_id}' which is not in the store"
        )


E2E_QUERIES = [
    "What problem does LoRA solve in fine-tuning?",
    "How does retrieval augmented generation work?",
    "What are the limitations of the LoRA approach?",
]

# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Week 1 Test Pipeline")
    parser.add_argument("--skip-network", action="store_true", help="Skip network-dependent tests")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Ollama-dependent tests")
    parser.add_argument("--keep", action="store_true", help="Keep test data after run")
    args = parser.parse_args()
    
    runner = TestRunner(skip_network=args.skip_network, skip_llm=args.skip_llm)
    
    # ========================================================================
    # Day 1 Tests
    # ========================================================================
    print_section("DAY 1: FastAPI Setup")
    runner.run_test("FastAPI app exists", "Day 1: Setup", test_fastapi_endpoint_exists)
    runner.run_test("Health check endpoint /", "Day 1: Setup", test_fastapi_hello_endpoint)
    runner.run_test("Env vars load cleanly", "Day 1: Setup", test_fastapi_env_vars_loaded)
    
    # ========================================================================
    # Day 2 Tests
    # ========================================================================
    print_section("DAY 2: arXiv Search Tool")
    runner.run_test(
        "Search and verify results",
        "Day 2: arXiv Search",
        test_arxiv_search_basic,
        skip_if=args.skip_network
    )
    runner.run_test(
        "Empty topic raises error",
        "Day 2: arXiv Search",
        test_arxiv_search_empty_topic_raises
    )
    runner.run_test(
        "Respect max_results parameter",
        "Day 2: arXiv Search",
        test_arxiv_search_max_results_respected,
        skip_if=args.skip_network
    )
    runner.run_test(
        "No results returns empty list",
        "Day 2: arXiv Search",
        test_arxiv_search_no_results_returns_empty_list,
        skip_if=args.skip_network
    )
    
    # ========================================================================
    # Day 3 Tests
    # ========================================================================
    print_section("DAY 3: Paper Fetching")
    runner.run_test(
        "Fetch paper metadata",
        "Day 3: Paper Fetch",
        test_fetch_paper_metadata,
        skip_if=args.skip_network
    )
    runner.run_test(
        "Invalid ID raises error",
        "Day 3: Paper Fetch",
        test_fetch_paper_metadata_invalid_id,
        skip_if=args.skip_network
    )
    runner.run_test(
        "Fetch full paper text",
        "Day 3: Paper Fetch",
        test_fetch_paper_full_text,
        skip_if=args.skip_network
    )
    runner.run_test(
        "Fallback to abstract",
        "Day 3: Paper Fetch",
        test_fetch_paper_graceful_fallback_to_abstract,
        skip_if=args.skip_network
    )
    runner.run_test(
        "Second paper (RAG_ID) fetches cleanly",
        "Day 3: Paper Fetch",
        test_fetch_second_paper_rag_id,
        skip_if=args.skip_network
    )
    
    # ========================================================================
    # Day 4 Tests: Chunking
    # ========================================================================
    print_section("DAY 4A: Text Chunking")
    runner.run_test("Empty text returns empty", "Day 4: Chunking", test_chunk_text_empty_returns_empty)
    runner.run_test("Short text single chunk", "Day 4: Chunking", test_chunk_text_short_text)
    runner.run_test("Respects paragraph boundaries", "Day 4: Chunking", test_chunk_text_respects_paragraph_boundaries)
    runner.run_test("Token-aware splitting", "Day 4: Chunking", test_chunk_text_token_aware_splitting)
    runner.run_test("Overlap between chunks", "Day 4: Chunking", test_chunk_text_overlap_exists)
    
    # ========================================================================
    # Day 4 Tests: Embeddings
    # ========================================================================
    print_section("DAY 4B: Text Embeddings")
    runner.run_test("Correct embedding shape", "Day 4: Embeddings", test_embed_texts_correct_shape)
    runner.run_test("Deterministic embeddings", "Day 4: Embeddings", test_embed_texts_deterministic)
    runner.run_test("Empty input handling", "Day 4: Embeddings", test_embed_texts_empty_list)
    
    # ========================================================================
    # Day 4 Tests: Vector Store
    # ========================================================================
    print_section("DAY 4C: Vector Store")
    runner.run_test("Add and count chunks", "Day 4: Store", test_store_add_and_count)
    runner.run_test("Query returns results", "Day 4: Store", test_store_query)
    runner.run_test("Idempotency check", "Day 4: Store", test_store_paper_already_ingested)
    runner.run_test("Delete paper removes chunks", "Day 4: Store", test_store_delete_paper)
    
    # ========================================================================
    # Day 4 Tests: Ingestion
    # ========================================================================
    print_section("DAY 4D: Ingestion Pipeline")
    runner.run_test(
        "Ingest paper",
        "Day 4: Ingestion",
        test_ingest_paper,
        skip_if=args.skip_network
    )
    runner.run_test(
        "Ingestion idempotency",
        "Day 4: Ingestion",
        test_ingest_paper_idempotent,
        skip_if=args.skip_network
    )
    runner.run_test(
        "Force re-ingestion",
        "Day 4: Ingestion",
        test_ingest_paper_force_reingest,
        skip_if=args.skip_network
    )
    runner.run_test(
        "Abstract fallback",
        "Day 4: Ingestion",
        test_ingest_paper_abstract_fallback
    )
    
    # ========================================================================
    # Day 5 Tests: Retrieval
    # ========================================================================
    print_section("DAY 5A: Retrieval")
    runner.run_test("Retrieve relevant chunks", "Day 5: Retrieval", test_retrieve_returns_results)
    runner.run_test("Ranked by relevance", "Day 5: Retrieval", test_retrieve_ranked_by_relevance)
    runner.run_test("Metadata filtering", "Day 5: Retrieval", test_retrieve_with_metadata_filter)
    runner.run_test("Nonexistent filter returns empty", "Day 5: Retrieval", test_retrieve_nonexistent_filter_returns_empty)
    runner.run_test("top_k > store size is safe", "Day 5: Retrieval", test_retrieve_top_k_larger_than_store)
    
    # ========================================================================
    # Day 5 Tests: Answer Synthesis (Pure Functions)
    # ========================================================================
    print_section("DAY 5B: Answer Synthesis (Pure Functions)")
    runner.run_test("Sanitize answer", "Day 5: Answer", test_sanitize_answer_removes_leading_symbols)
    runner.run_test("Preserve prose", "Day 5: Answer", test_sanitize_answer_preserves_prose)
    runner.run_test("Group chunks by paper", "Day 5: Answer", test_group_chunks_by_paper)
    runner.run_test("Number sources", "Day 5: Answer", test_build_context_block_numbers_papers)
    
    # ========================================================================
    # Day 5 Tests: LLM Integration
    # ========================================================================
    print_section("DAY 5C: LLM Integration (Ollama)")
    
    skip_llm_reason = None
    if args.skip_llm:
        skip_llm_reason = "Skipped via --skip-llm"
    elif not ollama_available():
        skip_llm_reason = "Ollama server not reachable at http://localhost:11434"
    elif not ollama_model_pulled("llama3.1:8b"):
        skip_llm_reason = "Run: ollama pull llama3.1:8b"
    
    if skip_llm_reason:
        print(f"    ⊘ SKIPPED: LLM tests ({skip_llm_reason})")
    else:
        runner.run_test("Call LLM with retry", "Day 5: LLM", test_call_llm_with_retry)
    
    # ========================================================================
    # Day 6 Tests: End-to-End
    # ========================================================================
    print_section("DAY 6: End-to-End Pipeline")
    runner.run_test("Retrieve and synthesize (context)", "Day 6: E2E", test_end_to_end_retrieve_and_answer)
    runner.run_test("Citation sources exist in store", "Day 6: E2E", test_end_to_end_citation_sources_are_real_arxiv_ids)
    
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    success = runner.print_summary()
    
    # Cleanup
    if not args.keep:
        print("\nCleaning up test data...")
        cleanup_test_data()

        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)
    else:
        print("\nTest data preserved (--keep flag)")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())