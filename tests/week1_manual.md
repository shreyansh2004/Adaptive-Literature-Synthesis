# Week 1 Testing & Component Documentation

This document explains the Week 1 test pipeline, component architecture, and what each piece does.

---

## Quick Start

### Run All Tests
```bash
python scripts/run_week1_tests.py
```

### Run With Options
```bash
python scripts/run_week1_tests.py --skip-network  # Skip arXiv tests (no internet needed)
python scripts/run_week1_tests.py --skip-llm      # Skip Ollama tests
python scripts/run_week1_tests.py --keep          # Keep test data after run (for inspection)
```

### Requirements
- **Core**: Python 3.8+, dependencies from `requirements.txt`
- **Network tests**: Internet access to export.arxiv.org (for live arXiv queries)
- **LLM tests** (optional): Local Ollama server with `llama3.1:8b` pulled
  ```bash
  ollama serve        # in one terminal
  ollama pull llama3.1:8b  # in another
  ```

---

## Test Pipeline Overview

The test suite is organized by **Day** (from the 4-week plan) and **Component**:

```
DAY 1: FastAPI Setup
├── Health check endpoint
└── FastAPI app structure

DAY 2: arXiv Search Tool (tools/arxiv_search.py)
├── Basic search and structure validation
├── Empty topic handling
├── max_results parameter
└── No-results handling

DAY 3: Paper Fetching (tools/paper_fetch.py)
├── Metadata fetching
├── Full PDF text extraction
├── Error handling for invalid papers
└── Abstract fallback

DAY 4: RAG Ingestion Pipeline
├── Chunking (rag/chunking.py)
│   ├── Empty input handling
│   ├── Paragraph-aware splitting
│   ├── Token-aware splitting for oversized text
│   └── Overlap verification
├── Embeddings (rag/embeddings.py)
│   ├── Correct shape validation
│   ├── Determinism check
│   └── Empty input handling
├── Vector Store (rag/store.py)
│   ├── Add/count operations
│   ├── Query and retrieval
│   ├── Idempotency (paper_already_ingested)
│   └── Delete operations
└── Ingestion (rag/ingest.py)
    ├── Single paper ingestion
    ├── Idempotency (no re-ingest without force)
    ├── Force re-ingestion
    └── Abstract fallback when full text unavailable

DAY 5: Retrieval & Synthesis
├── Retrieval (rag/retrieve.py)
│   ├── Chunk retrieval
│   ├── Relevance ranking
│   └── Metadata filtering
├── Answer Synthesis (rag/answer.py) - Pure Functions
│   ├── Sanitizing LLM output
│   ├── Grouping chunks by paper
│   └── Building context blocks
└── LLM Integration (Ollama)
    └── LLM calls with retry logic

DAY 6: End-to-End Integration
└── Full pipeline: retrieve → synthesize → answer
```

---

## Component Deep Dive

### Day 1: FastAPI Setup (`api/main.py`)

**Purpose**: Provide an HTTP entrypoint for the research assistant.

**What It Tests**:
- Can we import the FastAPI app without errors?
- Does the `/` endpoint return a valid response?

**Current Implementation**:
```python
@app.get("/")
def hello():
    return {"message": "FastAPI is working"}
```

**Why It Matters**: This is the foundation for Week 4's `/chat` endpoint. FastAPI handles async, concurrency, and OpenAPI docs automatically.

---

### Day 2: arXiv Search Tool (`tools/arxiv_search.py`)

**Purpose**: Query arXiv API and return structured paper metadata.

**What It Does**:
1. Takes a search topic (e.g., "LoRA fine-tuning")
2. Calls `export.arxiv.org/api/query` with parameters
3. Parses Atom feed response using `feedparser`
4. Returns list of `ArxivPaper` dataclass objects

**Input**: Search topic (string)
**Output**: `List[ArxivPaper]` with:
- `arxiv_id`: e.g., "2106.09685"
- `title`, `abstract`, `authors`
- `url`: Browse page (https://arxiv.org/abs/...)
- `pdf_url`: Direct PDF link
- `published`, `updated`: ISO date strings

**What Tests Check**:
- ✓ Valid result structure (all fields populated)
- ✓ Empty search raises `ValueError`
- ✓ `max_results` parameter honored
- ✓ Nonsense queries return empty list (no server error)

**Network Dependency**: Yes (calls `export.arxiv.org`)

---

### Day 3: Paper Fetching (`tools/paper_fetch.py`)

**Purpose**: Fetch full paper metadata and optionally extract PDF full text.

**Key Functions**:

#### `fetch_paper_metadata(arxiv_id: str) -> ArxivPaper`
- Queries arXiv API by ID (more targeted than search)
- Returns metadata only

#### `fetch_pdf_full_text(pdf_url: str) -> str`
- Downloads PDF and extracts text using `pypdf`
- Returns full text, raises on PDF parsing failure
- Non-fatal in the pipeline (abstract is used as fallback)

#### `fetch_paper(arxiv_id: str, include_full_text=True) -> FetchedPaper`
- Combines both above functions
- Returns `FetchedPaper` with optional `full_text` and `fetch_error` fields
- Tracks whether full text was successfully fetched

**What Tests Check**:
- ✓ Valid metadata returned for known paper
- ✓ Invalid ID raises error
- ✓ Full text extraction produces substantial output (>1000 chars)
- ✓ Abstract always available (fallback path)

**Network Dependency**: Yes (calls arXiv API + downloads PDFs)

---

### Day 4: RAG Ingestion Pipeline

#### Part A: Chunking (`rag/chunking.py`)

**Purpose**: Split research papers into manageable chunks for embedding.

**Strategy**:
1. Split by paragraph (`\n\n` boundary)
2. Accumulate paragraphs until reaching `chunk_size` tokens
3. If a single paragraph exceeds `chunk_size`, split it by tokens with overlap

**Configuration**:
- `CHUNK_SIZE_TOKENS = 500` (typical paper chunk ~350-500 tokens)
- `CHUNK_OVERLAP_TOKENS = 50` (context carryover between chunks)

**Why This Matters**:
- **Too large chunks**: Miss specificity during retrieval
- **Too small chunks**: Lose context, more expensive embeddings
- **Overlap**: Ensures semantic coherence at chunk boundaries

**What Tests Check**:
- ✓ Empty input yields empty output
- ✓ Short text stays in one chunk
- ✓ Paragraph boundaries respected (paragraphs not split unless oversized)
- ✓ Oversized paragraphs are token-split
- ✓ Overlapping tokens present between consecutive chunks

---

#### Part B: Embeddings (`rag/embeddings.py`)

**Purpose**: Convert text chunks into dense vector representations for similarity search.

**Current Implementation**: Local model (no API calls)
- Model: `BAAI/bge-small-en-v1.5` (from Hugging Face, via sentence-transformers)
- Embedding dimension: 384
- Downloaded on first run, cached locally

**Alternative Available**: Switch to OpenAI (`text-embedding-3-small`)
- Set `USE_OPENAI = True` in the code
- Better quality but requires `OPENAI_API_KEY` env var

**What Tests Check**:
- ✓ Correct output shape (e.g., `List[List[float]]`)
- ✓ Same input produces same output (deterministic)
- ✓ Empty input handled gracefully

---

#### Part C: Vector Store (`rag/store.py`)

**Purpose**: Persist and query embeddings using ChromaDB.

**What It Does**:
- Wraps ChromaDB's Python API
- Stores chunks as `(id, text, embedding, metadata)` tuples
- Supports cosine similarity search

**Key Methods**:
```python
add_chunks(ids, texts, embeddings, metadatas)     # Upsert chunks
query(query_embedding, top_k, where=None)         # Retrieve top-k
paper_already_ingested(arxiv_id) -> bool          # Idempotency check
delete_paper(arxiv_id)                            # Clear stale data
count() -> int                                     # Collection size
```

**Storage**:
- Location: `data/chroma/` (persistent, survives restarts)
- Collection: "papers" (can have multiple collections)
- Index: HNSW (Hierarchical Navigable Small World) with cosine distance

**What Tests Check**:
- ✓ Chunks can be added and counted
- ✓ Queries return results ranked by similarity
- ✓ Idempotency check prevents duplicate ingestion
- ✓ Delete operations work

---

#### Part D: Ingestion Pipeline (`rag/ingest.py`)

**Purpose**: Orchestrate the full pipeline: fetch → chunk → embed → store.

**Key Functions**:

#### `clean_paper_text(text: str) -> str`
- Removes references section (often low-value for QA)
- Strips header/author blocks that PDF extraction leaves behind

#### `ingest_paper(paper: Dict, store: VectorStore, force=False) -> int`
- Takes a paper loaded from JSON
- Prefers full text if available, falls back to abstract
- Returns number of chunks ingested
- **Idempotency**: Skips if paper already in store (unless `force=True`)

#### `ingest_all_raw_papers(raw_dir, force=False) -> int`
- Batch ingests every `.json` file in `data/raw/`
- Useful for bulk loading a corpus

**What Tests Check**:
- ✓ Single paper ingestion produces chunks
- ✓ Re-ingesting same paper without force returns 0 (idempotent)
- ✓ With `force=True`, re-ingestion happens
- ✓ Abstract fallback works when full text unavailable

---

### Day 5: Retrieval & Synthesis

#### Part A: Retrieval (`rag/retrieve.py`)

**Purpose**: Given a query, find the most relevant chunks from the vector store.

**Key Function**:
```python
retrieve(query: str, store: VectorStore, top_k=5, where=None) -> List[RetrievedChunk]
```

**Process**:
1. Embed the query using the **same model** as ingestion (critical!)
2. Query the vector store for top-k nearest neighbors
3. Wrap results in `RetrievedChunk` dataclass with metadata

**Metadata Per Chunk**:
- `arxiv_id`, `title`, `authors`, `published`, `url`
- `chunk_index`: Which chunk within the paper
- `source`: "abstract" or "full_text"
- `distance`: Cosine distance (lower = more similar)
- `score`: 1 - distance (higher = more similar)

**What Tests Check**:
- ✓ Retrieval returns non-empty results
- ✓ Results ranked by distance (ascending, most relevant first)
- ✓ Metadata filtering works (e.g., only chunks from LORA_ID)

---

#### Part B: Answer Synthesis - Pure Functions (`rag/answer.py`)

**Purpose**: Process retrieved chunks and prepare context for LLM.

**Key Functions**:

#### `sanitize_answer(text: str) -> str`
- Problem: Some LLMs leak raw equations/notation before actual prose
- Solution: Strip leading non-prose lines (symbols, equations, notations)
- Heuristic: A "real" line is >40 chars and >60% alphabetic
- Safety: Returns original if no clean line found

#### `group_chunks_by_paper(chunks: List[RetrievedChunk]) -> OrderedDict[arxiv_id, List[RetrievedChunk]]`
- Groups retrieved chunks by their source paper
- Preserves order of first appearance (best relevance rank)
- Prevents jumbling together chunks from different papers

#### `build_context_block(grouped: OrderedDict) -> str`
- Formats grouped chunks as a numbered context for the LLM
- Example output:
  ```
  [1] Source: 2106.09685 — LoRA: Low-Rank Adaptation...
  Authors: ...
  ---
  <chunk 1 text>
  ---
  <chunk 2 text>
  
  [2] Source: 2005.11401 — Retrieval-Augmented Generation...
  ...
  ```
- Numbers by **paper**, not by chunk

**What Tests Check**:
- ✓ Leading equations stripped
- ✓ Clean prose preserved
- ✓ Chunks grouped by paper
- ✓ Sources numbered by paper (not chunk)

---

#### Part C: LLM Integration (`rag/answer.py`)

**Purpose**: Call local Ollama to synthesize answers from context.

**Key Function**:
```python
_call_llm_with_retry(messages, max_retries=2, base_wait=5) -> dict
```

**Details**:
- Target: `http://localhost:11434/api/chat`
- Model: `llama3.1:8b` (configurable)
- Retry logic: 2 retries with exponential backoff (5s, 10s)
- Timeout: 240 seconds (allows for slow local inference)

**Response Format** (Ollama's shape, not OpenAI):
```json
{
  "message": {
    "role": "assistant",
    "content": "LoRA is a technique that..."
  }
}
```

**System Prompt**: Enforces:
- Only use provided excerpts (no outside knowledge)
- Cite sources with `[n]` notation
- Don't merge findings from different papers
- No raw text/equations copied; use plain language
- Start directly with prose (no leading symbols)

**What Tests Check**:
- ✓ Ollama server is reachable
- ✓ Model is pulled
- ✓ Response has correct shape

---

### Day 6: End-to-End Integration

**Purpose**: Verify the full pipeline works end-to-end: query → retrieve → synthesize.

**What Tests Check**:
- ✓ Retrieval returns relevant chunks
- ✓ Chunks are properly grouped and formatted
- ✓ Context block has numbered sources

---

## Test Isolation & Cleanup

All tests use separate test directories to avoid polluting your real data:

```
data/
├── raw/              ← YOUR real fetched papers
├── chroma/           ← YOUR real vector store
├── raw_test/         ← Tests only (auto-cleaned)
└── chroma_test/      ← Tests only (auto-cleaned)
```

**Cleanup Behavior**:
- Default: Test directories deleted after run
- `--keep`: Preserves `data/raw_test/` and `data/chroma_test/` for inspection

---

## Common Issues & Troubleshooting

### "Ollama server not reachable at http://localhost:11434"
**Solution**: Start Ollama in a separate terminal:
```bash
ollama serve
```

### "Model 'llama3.1:8b' not found in `ollama list`"
**Solution**: Pull the model:
```bash
ollama pull llama3.1:8b
```

### Tests pass but you want to verify retrieval quality
**Solution**: Check `data/raw_test/` and `data/chroma_test/` with `--keep`:
```bash
python scripts/run_week1_tests.py --keep
# Now inspect the test data
python -c "from rag.retrieve import retrieve; from rag.store import VectorStore; \
store = VectorStore('data/chroma_test', 'papers_test'); \
print(retrieve('LoRA fine-tuning', store, top_k=3))"
```

### Network tests failing (no internet available)
**Solution**: Skip network tests:
```bash
python scripts/run_week1_tests.py --skip-network
```

---

## Test Results Interpretation

After running tests, you'll see a summary like:

```
================================================================================
TEST SUMMARY
================================================================================
  ✓ Day 1: Setup: 2/2 (100%)
  ✓ Day 2: arXiv Search: 4/4 (100%)
  ✓ Day 3: Paper Fetch: 4/4 (100%)
  ✓ Day 4: Chunking: 5/5 (100%)
  ✓ Day 4: Embeddings: 3/3 (100%)
  ✓ Day 4: Store: 4/4 (100%)
  ✓ Day 4: Ingestion: 4/4 (100%)
  ✓ Day 5: Retrieval: 3/3 (100%)
  ✓ Day 5: Answer: 4/4 (100%)
  ✓ Day 6: E2E: 1/1 (100%)

Overall: 38/38 passed (100%)
```

**What This Means**:
- ✓ All core RAG components are working
- You can ask questions and get cited answers
- Ready to wrap in LangGraph (Week 2)

---

## Component Relationships

```
┌─────────────────┐
│  arXiv API      │
└────────┬────────┘
         │
    ┌────▼─────────┐
    │ arxiv_search │
    └────┬─────────┘
         │
    ┌────▼───────────┐
    │ paper_fetch    │
    └────┬───────────┘
         │ (fetches full text + metadata)
         │
    ┌────▼──────────────────────────────────┐
    │        data/raw/ (JSON storage)        │
    └────┬──────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────────┐
    │  ingest_paper                                    │
    │  ┌──────────┐  ┌─────────────┐  ┌────────────┐  │
    │  │ chunking │→ │ embeddings  │→ │ vector     │  │
    │  │          │  │             │  │ store      │  │
    │  └──────────┘  └─────────────┘  └────────────┘  │
    └────┬────────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   data/chroma/ (ChromaDB storage)  │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────┐
    │ retrieve      │
    │ (query→top-k) │
    └────┬──────────┘
         │
    ┌────▼──────────────────────────┐
    │ answer synthesis              │
    │ (group→format→prompt LLM)      │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────┐
    │ Ollama LLM    │
    └───────────────┘
         │
      (answer with citations)
```

---

## Next Steps (Week 2)

Once Week 1 tests pass, Week 2 wraps this in a **LangGraph state machine**:

```python
# Week 2: Agentic Layer
graph = StateGraph(AgentState)

graph.add_node("planner", plan_next_action)      # Decide: search/retrieve/answer?
graph.add_node("searcher", search_arxiv_node)    # Search arXiv
graph.add_node("retriever", retrieve_node)       # Retrieve from local store
graph.add_node("synthesizer", synthesizer_node)  # Answer + cite

graph.add_conditional_edges(
    "planner",
    route_decision,
    {
        "search_arxiv": "searcher",
        "retrieve_local": "retriever",
        "answer": "synthesizer",
    }
)
```

The agent will autonomously decide which tools to use, rather than you hardcoding the path.

---

## Testing Best Practices

1. **Run tests before making changes**: Establish a baseline
2. **Run tests after making changes**: Catch regressions
3. **Use `--keep` to debug failures**: Inspect test data
4. **Skip network/LLM if needed**: Focus on core logic
5. **Add new tests as you add features**: Growing your safety net

---

## Summary Table: What Each Day Tests

| Day | Component | Tests | Status |
|-----|-----------|-------|--------|
| 1 | FastAPI Setup | 2 tests | ✓ Complete |
| 2 | arXiv Search | 4 tests | ✓ Complete |
| 3 | Paper Fetch | 4 tests | ✓ Complete |
| 4a | Chunking | 5 tests | ✓ Complete |
| 4b | Embeddings | 3 tests | ✓ Complete |
| 4c | Vector Store | 4 tests | ✓ Complete |
| 4d | Ingestion | 4 tests | ✓ Complete |
| 5a | Retrieval | 3 tests | ✓ Complete |
| 5b | Answer Synthesis | 4 tests | ✓ Complete |
| 5c | LLM Integration | 1 test | ✓ Complete |
| 6 | End-to-End | 1 test | ✓ Complete |
| **Total** | | **38 tests** | **✓ Complete** |

---

## Running Tests in CI/CD

Example GitHub Actions workflow (optional):

```yaml
name: Week 1 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python scripts/run_week1_tests.py --skip-network --skip-llm
```

This runs all tests *except* those needing network/Ollama, making CI fast and reliable.
