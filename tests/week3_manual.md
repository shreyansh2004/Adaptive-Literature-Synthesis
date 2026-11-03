# Week 3 Manual Testing Guide — Memory + MCP

This guide is the Week 3 companion to the Week 1 and Week 2 testing docs. It focuses on the new capabilities introduced in Days 15–20: persistent memory, memory-aware routing, session summarization, and MCP-based arXiv search.

---

## Quick Start

### Prerequisites
- Python environment activated
- Local Ollama server available for planner/synthesizer paths if you want full end-to-end behavior
- Network access for live arXiv/MCP tests

### Run Automated Tests
```bash
python tests/week3_testing.py
```

### Skip Network Tests
```bash
python tests/week3_testing.py --skip-network
```

---

## What Week 3 Should Prove

By the end of Week 3, the system should be able to:
1. Persist information across sessions
2. Recall papers seen earlier
3. Inject memory into planner context
4. Summarize a session for future recall
5. Expose at least one tool through MCP instead of direct Python import

---

## Test Categories

### 1. Memory Persistence

#### Scenario 1.1 — Paper is stored and reloaded
```python
store = MemoryStore(db_path="/tmp/memory.db")
store.start_session("s1")
store.record_paper_read("2106.09685", "LoRA paper", "https://arxiv.org/abs/2106.09685")
store.end_session()

reloaded = MemoryStore(db_path="/tmp/memory.db")
assert reloaded.has_read_paper("2106.09685")
```

Expected:
- The paper is still present after a fresh MemoryStore instance is created.
- `get_paper("2106.09685")` returns a populated record.

#### Scenario 1.2 — Topics and conclusions survive a new session
```python
store.start_session("s2")
store.record_topic("LoRA fine-tuning")
store.record_conclusion("LoRA lowers trainable parameters", topic="LoRA fine-tuning")
```

Expected:
- The topic appears in `get_memory_snapshot().recent_topics`
- The conclusion appears in `get_memory_snapshot().recent_conclusions`

---

### 2. Memory Read/Write Flow

#### Scenario 2.1 — Session summarizer writes structured memory
Use a fake LLM that returns JSON with:
- `summary`
- `topics`
- `conclusions`

Expected:
- The store ends the session with a non-empty summary.
- The topics and conclusions are persisted into the database.

#### Scenario 2.2 — Context injection is human-readable
Call `format_memory_context(snapshot)` after recording a paper and a topic.

Expected:
- The output contains headings like `MEMORY FROM PAST SESSIONS:`
- It lists known paper IDs and topics
- It is short enough to be injected as planner context

---

### 3. Memory-Aware Routing

#### Scenario 3.1 — Repeated paper query uses cached memory
User asks about a paper already seen earlier.

Expected behavior:
- `decide_next_action()` returns `use_cached_paper`
- The planner surfaces a notice such as:
  `I've already read paper 2106.09685 in a previous session...`

#### Scenario 3.2 — Unknown paper still falls through
Ask about a fresh paper ID not present in memory.

Expected:
- `decide_next_action()` returns `fall_through`
- The system behaves like a normal first-time lookup

---

### 4. MCP Integration

#### Scenario 4.1 — MCP search returns paper metadata
Use the MCP client:
```python
from tools.arxiv_mcp_client import search_arxiv_via_mcp
papers = search_arxiv_via_mcp(topic="LoRA fine-tuning", max_results=2)
```

Expected:
- Returns a list of paper dictionaries
- Each result includes at least:
  - `arxiv_id`
  - `title`
  - `abstract`
  - `authors`
  - `url`
  - `pdf_url`

#### Scenario 4.2 — MCP server is discoverable as a tool
The MCP server should expose the `search_arxiv_tool` tool through the stdio transport.

Expected:
- The client can initialize the session successfully
- The tool call returns structured results without direct Python imports

---

## Manual End-to-End Flow

Use this sequence to validate the whole Week 3 experience manually:

1. Start a new session and ask about a paper such as `2106.09685`.
2. Let the system store the paper in memory.
3. Start a new session and ask again about the same paper.
4. Confirm that the agent mentions it has already read that paper before.
5. Confirm that it does not re-fetch it unnecessarily.
6. Optionally check that a topic summary was stored for later sessions.

---

## Exit Criteria for Week 3

You can consider Week 3 complete when:
- A paper seen in one session is recognized in a later session
- Memory context is actually injected into the planner path
- A session summary is persisted for future use
- At least one tool is reachable through MCP rather than direct import
