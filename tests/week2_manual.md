# Week 2 Manual Testing Guide — Day 13 (Stress-Testing)

Complete guide for manually testing the LangGraph agent with various query types and verifying routing logic.

---

## Quick Start

### Prerequisites
- Ollama running: `ollama serve` (in separate terminal)
- Model pulled: `ollama pull llama3.1:8b`
- Week 1 complete: ChromaDB with ingested papers, embeddings working

### Run Automated Tests
```bash
# All tests
python tests/week2_testing.py

# Skip Ollama (if server not available)
python tests/week2_testing.py --skip-llm

# Single query
python tests/week2_testing.py --query "What is LoRA?"

# Verbose output
python tests/week2_testing.py --verbose
```

---

## Test Categories & Expected Behavior

### Category 1: Planner Routing Logic

**What's Being Tested**: Does the planner make correct routing decisions?

#### Test 1.1: Empty State → Answer Now
```python
Query: "What is LoRA?"
Initial State: retrieved_docs=[], search_results=[]

Expected Path: planner → answer
Expected Behavior:
  - Planner sees no docs, simple question
  - Decides "answer_now"
  - Routes directly to synthesizer
  - Synthesizer returns: "I couldn't find relevant information..."

✓ PASS if: next_action == "answer_now"
```

#### Test 1.2: Unknown Topic → Search ArXiv
```python
Query: "Tell me about the latest papers on quantum computing"
Initial State: retrieved_docs=[], search_results=[]

Expected Path: planner → search_arxiv → fetch_and_ingest → retrieve_local → planner → answer
Expected Behavior:
  - Planner sees unknown topic, no local docs
  - Decides "search_arxiv"
  - Fetches papers from arXiv
  - Ingests them into ChromaDB
  - Retrieves relevant chunks
  - Loops back to planner with populated retrieved_docs
  - Planner now decides "answer_now"
  - Synthesizer produces cited answer

✓ PASS if: 
  - Multiple tool_calls: [search_arxiv, fetch_and_ingest, retrieve_local, ...]
  - answer field is populated
  - sources field has 1+ entries
```

#### Test 1.3: Max Hops Enforced
```python
Query: "Some query"
Initial State: hop_count=4 (at limit)

Expected Path: planner (safety check) → answer
Expected Behavior:
  - Planner checks: if hop_count >= 4, force "answer_now"
  - Prevents infinite loops
  - Forces exit even if state isn't ready

✓ PASS if: next_action == "answer_now" and hop_count incremented
```

#### Test 1.4: No Duplicate Searches
```python
Query: "Tell me more about LoRA"
Initial State: search_results=[{paper1}, {paper2}]

Expected Path: planner → answer (NOT search_arxiv again)
Expected Behavior:
  - Planner sees search_results already populated
  - Code override: if action=="search_arxiv" and search_results exist → change to "answer_now"
  - Prevents wasteful re-searching

✓ PASS if: next_action != "search_arxiv"
```

#### Test 1.5: Compare Requires Keyword
```python
Query: "What do these papers discuss?"
Initial State: retrieved_docs=[{paper1}, {paper2}] (2 papers)

Expected Path: planner → answer (NOT compare_papers)
Expected Behavior:
  - Even with 2+ distinct papers present
  - Query doesn't explicitly ask for "comparison"
  - Code override: if action=="compare_papers" but query has no comparison keyword → change to "answer_now"
  - Prevents false positives

✓ PASS if: next_action != "compare_papers"
```

#### Test 1.6: Compare with Keyword + Multiple Papers
```python
Query: "Compare the approaches of LoRA and LoRA-FAIR"
Initial State: retrieved_docs=[{LoRA}, {LoRA-FAIR}] (2+ papers)

Expected Path: planner → compare_papers
Expected Behavior:
  - Query has "Compare" keyword
  - 2+ distinct papers present
  - Planner decides "compare_papers"
  - Routes to comparator node
  - Produces structured comparison (method, results, limitations)

✓ PASS if: next_action == "compare_papers"
```

---

### Category 2: Multi-Hop Flows

**What's Being Tested**: Can the agent chain multiple steps together?

#### Test 2.1: Full Search → Ingest → Retrieve → Answer Flow
```python
Query: "What are the latest papers on federated learning with LoRA?"

Step-by-Step Trace:
  [1] Planner → decides "search_arxiv"
      tool_calls: ["planner:search_arxiv"]
  
  [2] Search ArXiv → finds papers
      tool_calls: [..., "search_arxiv(5)"]  # 5 results
      Populates: state["search_results"]
  
  [3] Fetch & Ingest → downloads, chunks, embeds
      tool_calls: [..., "fetch_and_ingest(new=3, cached=0, failed=0, chunks_added=45)"]
      Populates: state["tool_calls"] (for tracking)
  
  [4] Retrieve Local → queries ChromaDB
      tool_calls: [..., "retrieve_local(5)"]  # top 5 chunks
      Populates: state["retrieved_docs"]
  
  [5] Planner (2nd time) → sees retrieved_docs now filled
      tool_calls: [..., "planner:answer_now"]
      Decides: "answer_now"
  
  [6] Synthesizer → generates cited answer
      tool_calls: [..., "synthesizer(3 papers)"]
      Final: state["answer"] + state["sources"]

Expected Metrics:
  - hop_count: 2 (planner called twice)
  - tool_calls length: 6+
  - answer length: > 100 chars
  - sources length: >= 1

✓ PASS if: All steps execute without error
```

#### Test 2.2: Quick Retrieve → Answer (Local Docs)
```python
Query: "What does the LoRA paper say about parameter efficiency?"

Assumes: LoRA paper already ingested in ChromaDB

Step-by-Step Trace:
  [1] Planner → checks state
      No search_results, but might have local docs
      Decides: "retrieve_local" (if planner learns from context)
      OR "answer_now" with fallback
  
  [2] Retrieve Local → finds relevant chunks
      tool_calls: [..., "retrieve_local(5)"]
      Populates: state["retrieved_docs"]
  
  [3] Planner (2nd time) or directly Synthesizer
      Decides: "answer_now"
  
  [4] Synthesizer → cites paper
      Produces answer with [1] reference

✓ PASS if: answer contains specific LoRA details + [1] citation
```

---

### Category 3: Answer Synthesis

**What's Being Tested**: Does the answer have proper format, citations, and sources?

#### Test 3.1: Citations in Answer
```python
Query: "What is overfitting?"
retrieved_docs: [
  {
    "text": "Overfitting occurs when a model memorizes training data...",
    "arxiv_id": "9999.99999",
    "title": "Paper on Generalization",
    ...
  }
]

Expected Answer Format:
  "Overfitting is a phenomenon that occurs when... [1]"
  
  Where [1] references the first (and only) source

Checking:
  - answer contains "[1]" or similar citation marker
  - sources[0] has arxiv_id, title, url
  - answer NOT empty

✓ PASS if: answer has "[n]" citation(s)
```

#### Test 3.2: Graceful Handling of No Docs
```python
Query: "Some totally obscure topic"
retrieved_docs: []

Expected Answer:
  "I couldn't find any relevant information for this question, either locally or via search."
  
  OR similar message indicating lack of information

NOT:
  - Hallucinated facts
  - Blank answer
  - Error traceback

✓ PASS if: answer contains "couldn't find" or "no information"
```

#### Test 3.3: Sources Metadata
```python
For any answer with retrieved_docs:

Expected sources structure:
  {
    "n": 1,
    "title": "Paper Title",
    "url": "http://arxiv.org/abs/...",
    "arxiv_id": "2106.09685"
  }

✓ PASS if: Each source has n, title, url, arxiv_id
```

---

### Category 4: Paper Comparison

**What's Being Tested**: Can the agent compare 2+ papers with structured output?

#### Test 4.1: Structured Comparison
```python
Query: "Compare LoRA-FAIR and Federated Sketching LoRA"
retrieved_docs: [
  {
    "text": "LoRA-FAIR uses server-side aggregation bias correction...",
    "arxiv_id": "2411.14961",
    "title": "LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation..."
  },
  {
    "text": "FSLoRA uses sketching to compress LoRA updates...",
    "arxiv_id": "2501.19389",
    "title": "Federated Sketching LoRA: A Flexible Framework..."
  }
]

Expected Output (JSON-like structure):
  [1] Method: LoRA-FAIR introduces a server-side aggregation bias correction...
      Results: Addresses inconsistencies in heterogeneous federated settings
      Limitations: Not stated in excerpts
  
  [2] Method: FSLoRA uses sketching mechanism for compression...
      Results: Reduces communication overhead
      Limitations: Not stated in excerpts
  
  Key differences: LoRA-FAIR focuses on aggregation bias correction, 
                   while FSLoRA focuses on communication efficiency.

Checking:
  - Both papers numbered [1] and [2]
  - Method/Results/Limitations sections present
  - Key differences summarized

✓ PASS if: answer contains [1], [2], and comparison text
```

#### Test 4.2: Comparison Requires 2+ Papers
```python
Query: "Compare these papers"
retrieved_docs: [
  {
    "arxiv_id": "1111.11111",
    "title": "Single Paper"
  }
]

Expected Output:
  "I need at least 2 distinct papers already retrieved to compare — 
   try asking me to search or retrieve relevant papers first."

✓ PASS if: answer rejects with "need ... papers" message
```

---

### Category 5: State Management & Edge Cases

**What's Being Tested**: Does the agent properly track state across hops?

#### Test 5.1: Hop Count Increments
```python
Any multi-step query:

Expected:
  - Initial: hop_count = 0
  - After 1st planner call: hop_count = 1
  - After 2nd planner call: hop_count = 2
  - etc.

✓ PASS if: result["hop_count"] > initial value
```

#### Test 5.2: Tool Calls Accumulated
```python
Query: "What is LoRA?"

Expected:
  tool_calls = [
    "planner:answer_now",
    "synthesizer(0 papers)"
  ]
  
  OR (if multi-hop):
  tool_calls = [
    "planner:search_arxiv",
    "search_arxiv(5)",
    "fetch_and_ingest(new=1, cached=0, failed=0, chunks_added=30)",
    "retrieve_local(5)",
    "planner:answer_now",
    "synthesizer(1 papers)"
  ]

✓ PASS if: len(tool_calls) > 0 and all entries are strings
```

#### Test 5.3: Query Preserved Through Hops
```python
Original Query: "What papers discuss federated learning with LoRA?"

After all hops:
  result["query"] == "What papers discuss federated learning with LoRA?"
  (unchanged)

✓ PASS if: result["query"] == original query
```

---

## Example Interactive Sessions

### Session 1: Simple Question (1 hop)
```
$ python tests/week2_testing.py --query "What is LoRA?"

Q: What is LoRA?
  → tool_calls: ["planner:answer_now", "synthesizer(0 papers)"]
  → answer: "I couldn't find any relevant information for this question..."
  → sources: []
  → hop_count: 1
```

**Explanation**: 
- Planner sees empty state, no docs → decides "answer_now"
- Synthesizer has no retrieved_docs → returns "couldn't find" message
- Single planner call = 1 hop

---

### Session 2: Multi-Hop Search
```
$ python tests/week2_testing.py --query "What are recent papers on federated learning?"

Q: What are recent papers on federated learning?
  → tool_calls: [
      "planner:search_arxiv",
      "search_arxiv(5)",
      "fetch_and_ingest(new=3, cached=0, failed=0, chunks_added=87)",
      "retrieve_local(5)",
      "planner:answer_now",
      "synthesizer(3 papers)"
    ]
  → answer: "Recent work on federated learning has focused on..."
  → sources: [
      {"n": 1, "title": "...", "arxiv_id": "2411.14961", "url": "..."},
      {"n": 2, "title": "...", "arxiv_id": "2501.19389", "url": "..."},
      {"n": 3, "title": "...", "arxiv_id": "2410.20625", "url": "..."}
    ]
  → hop_count: 2
```

**Explanation**:
- Planner sees empty state → decides "search_arxiv"
- Searches arXiv, finds 5 papers, downloads 3
- Ingests into ChromaDB (87 chunks total)
- Retrieves top-5 chunks from those papers
- Loops back to planner with populated retrieved_docs
- Planner now decides "answer_now"
- Synthesizer generates answer with 3 sources
- Two planner calls = 2 hops

---

### Session 3: Paper Comparison
```
$ python tests/week2_testing.py --query "Compare LoRA-FAIR and Federated Sketching LoRA"

Q: Compare LoRA-FAIR and Federated Sketching LoRA
  → Initial retrieval: 2 papers found (from previous sessions or new search)
  → tool_calls: [
      "planner:compare_papers",
      "comparator(2 papers)"
    ]
  → answer: "[1] Method: LoRA-FAIR uses server-side aggregation bias correction...
             [2] Method: Federated Sketching LoRA uses sketching for compression...
             Key differences: LoRA-FAIR addresses aggregation heterogeneity while 
                              FSLoRA addresses communication efficiency."
  → sources: [
      {"n": 1, "title": "LoRA-FAIR...", "arxiv_id": "2411.14961", ...},
      {"n": 2, "title": "Federated Sketching LoRA...", "arxiv_id": "2501.19389", ...}
    ]
  → hop_count: 1
```

**Explanation**:
- Query explicitly asks to "Compare"
- Papers already in retrieved_docs (from prior session or fresh retrieval)
- Planner directly decides "compare_papers"
- Comparator produces structured comparison
- Single decision = 1 hop

---

## Troubleshooting

### Problem: "Ollama server not reachable"
**Solution**: 
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Verify model
ollama list

# If model missing:
ollama pull llama3.1:8b
```

### Problem: Tests pass but answers are generic
**Root Cause**: Likely no papers in ChromaDB yet

**Solution**:
```bash
# Verify Week 1 papers are ingested:
python -c "from rag.store import VectorStore; s = VectorStore(); print(f'Total chunks: {s.count()}')"

# If 0, ingest papers:
python scripts/run_week1_tests.py
```

### Problem: Planner keeps routing to `answer_now` even for search queries
**Root Cause**: 
- No papers in ChromaDB → planner sees no local docs but decides "answer_now" instead of searching
- Planner model (llama3.1:8b) might be underfitting

**Solution**:
- Populate ChromaDB first (see above)
- Check planner prompt in `agents/planner.py`

### Problem: Comparator returns error or incomplete JSON
**Root Cause**: 
- Llama might struggle with JSON formatting
- Less than 2 papers in retrieved_docs

**Solution**:
```bash
# Test with explicit 2+ papers:
python tests/week2_testing.py --query "Compare LoRA-FAIR and Federated Sketching LoRA"

# If still failing, check:
# 1. Are papers actually retrieved? (check retrieved_docs size)
# 2. Is Ollama returning valid JSON? (check raw response in comparator.py debug print)
```

### Problem: Infinite loop / max hops hit
**Expected**: If max_hops=4 is hit, should force `answer_now`

**If looping**:
- Bug in conditional routing
- Check `graph.add_conditional_edges` routing logic
- Ensure all node returns include necessary state updates

---

## Verification Checklist (Before Calling Week 2 "Complete")

- [✓] Test 1.1 passes (empty state → answer)
- [✓] Test 1.2 passes (unknown topic → search)
- [✓] Test 1.3 passes (max hops enforced)
- [✓] Test 1.4 passes (no duplicate search)
- [✓] Test 1.5 passes (compare needs keyword)
- [✓] Test 1.6 passes (compare with keyword works)
- [✓] Test 2.1 passes (full search→ingest→retrieve→answer)
- [✓] Test 2.2 passes (quick retrieve→answer)
- [✓] Test 3.1 passes (citations in answer)
- [✓] Test 3.2 passes (no docs → "couldn't find")
- [✓] Test 3.3 passes (sources metadata)
- [✓] Test 4.1 passes (structured comparison)
- [✓] Test 4.2 passes (comparison requires 2+ papers)
- [✓] Test 5.1 passes (hop count increments)
- [✓] Test 5.2 passes (tool calls accumulated)
- [✓] Test 5.3 passes (query preserved)

If all pass: **Week 2 Complete!** ✓

---

## Next Steps (Week 3)

- [ ] Add memory system (`memory/schema.py`, `memory/store.py`)
- [ ] Integrate memory into planner context
- [ ] Set up MCP server for arxiv_search
- [ ] Test cross-session memory retention

---

## Reference: Agent Flow Diagram

```
User Query
    ↓
[Planner Node]
    ├─→ search_arxiv?
    │    ├─→ [Search ArXiv]
    │    ├─→ [Fetch & Ingest]
    │    ├─→ [Retrieve Local]
    │    └─→ [Back to Planner]
    │
    ├─→ retrieve_local?
    │    ├─→ [Retrieve Local]
    │    └─→ [Back to Planner]
    │
    ├─→ answer_now?
    │    └─→ [Synthesizer Node]
    │         └─→ [Answer + Sources]
    │
    └─→ compare_papers?
         └─→ [Comparator Node]
             └─→ [Structured Comparison]

[Safety Valve]
If hop_count >= 4 → Force answer_now
```

---

## File Locations

- **Agent Code**: `agents/graph.py`, `agents/planner.py`, `agents/tool_nodes.py`, `agents/synthesizer.py`, `agents/comparator.py`
- **Test Suite**: `tests/week2_testing.py`
- **Manual Guide**: `tests/week2_manual.md` (this file)
- **Week 1 Code**: `rag/`, `tools/`
