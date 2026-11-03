# Agentic ML/AI Research Assistant

An agentic research assistant that answers questions over ML/AI papers by deciding for itself when to search arXiv, retrieve from a local corpus, compare papers, or answer from session memory.

Built with **LangGraph** (routing), **ChromaDB** (vector store), **Ollama** (local LLM), and **FastAPI**. One tool (`arxiv_search`) is served via **MCP**.

---

## What it does

Given a natural-language research question, the agent:

1. **Plans** the next action (search / retrieve / compare / answer)
2. Optionally **searches arXiv** via an MCP tool, fetches PDFs, and ingests them
3. **Retrieves** relevant chunks from a local Chroma store
4. **Synthesizes** a cited answer (or a structured comparison across papers)
5. **Remembers** papers read, topics covered, and session conclusions in SQLite

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LoRA fine-tuning?"}'
```

```json
{
  "session_id": "...",
  "answer": "LoRA fine-tuning ... [1]",
  "sources": [{"n": 1, "arxiv_id": "2106.09685", "title": "LoRA: Low-Rank Adaptation of Large Language Models", ...}],
  "tool_calls": ["planner:retrieve_local(prefer_local)", "retrieve_local(6)", "planner:answer_now", "synthesizer(1 papers)"],
  "memory_notice": null
}
```

---

## Architecture

```
                     ┌─────────────┐
          query ───► │   planner   │◄── memory snapshot (SQLite)
                     └──────┬──────┘
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     search_arxiv    retrieve_local    answer_now / compare_papers
            │               │               │
            ▼               │               ▼
   fetch_and_ingest         │         synthesizer / comparator
            │               │               │
            └──────► retrieve_local ◄───────┘
                            │
                            ▼
                      ChromaDB + Ollama
```

| Component | Role |
|-----------|------|
| `agents/planner.py` | Routes: search, retrieve, compare, answer; gibberish/self-ref gates |
| `agents/tool_nodes.py` | Wrappers for arXiv search, local retrieve, fetch+ingest |
| `agents/synthesizer.py` | Cited answer from retrieved chunks |
| `agents/comparator.py` | Multi-paper comparison |
| `agents/graph.py` | LangGraph state machine |
| `rag/` | Chunk → embed → store → retrieve |
| `memory/` | SQLite session memory (papers, topics, conclusions) |
| `mcp_server/` | MCP server exposing `search_arxiv` |
| `api/main.py` | FastAPI `/chat`, `/health`, `/sessions/{id}` |

**Models (defaults):**
- LLM: `llama3.1:8b` via Ollama
- Embeddings: `BAAI/bge-small-en-v1.5` (local `sentence-transformers`)

---

## Repo layout

```
research-assistant/
├── agents/           # LangGraph nodes + graph
├── api/              # FastAPI app
├── rag/              # ingest, retrieve, store, embeddings, chunking
├── memory/           # SQLite store, routing, session summarizer
├── tools/            # arxiv_search, paper_fetch, MCP client
├── mcp_server/       # MCP server process
├── eval/             # questions.json, run_eval.py, results*.csv
├── scripts/          # fetch_corpus.py: pin core papers
├── tests/            # week1-3 smoke tests
├── data/             # memory.db, chroma/, raw/ (gitignored)
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with `llama3.1:8b` pulled:
  ```bash
  ollama pull llama3.1:8b
  ```
- Network access for first-time model/paper downloads

---

## Setup

```bash
# Clone and enter
git clone <repo-url>
cd research-assistant

# Virtualenv
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Optional env vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEMORY_DB_PATH` | `data/memory.db` | SQLite memory DB |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama chat endpoint |
| `SUMMARIZER_MODEL` | `llama3.1:8b` | Session summarizer model |
| `EVAL_API_URL` | `http://localhost:8000` | Eval script target |
| `JUDGE_MODEL` | `llama3.1:8b` | Eval LLM-as-judge |

---

## Ingest a paper corpus

Pin and fetch core papers, then ingest into Chroma:

```bash
python scripts/fetch_corpus.py
python -m rag.ingest          # ingest every JSON in data/raw/ into Chroma
```

Or from a Python shell:

```python
from rag.ingest import ingest_all_raw_papers
ingest_all_raw_papers()
```

Papers land in `data/raw/{arxiv_id}.json`; vectors in `data/chroma/`.

---

## Run the API

```bash
# Ensure Ollama is running, then:
uvicorn api.main:api --reload --port 8000
```

### Endpoints

**`POST /chat`**

```json
{ "query": "What is LoRA fine-tuning?", "session_id": null }
```

- Omitting `session_id` starts a new session.
- Reuse `session_id` for follow-ups (`What did we just discuss?`).

**`GET /health`** → `{"status": "ok"}`

**`GET /sessions/{session_id}`** → recent topics, conclusions, known paper IDs

### Example multi-turn session

```bash
# Turn 1
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LoRA fine-tuning?"}' | tee /tmp/r1.json

# Turn 2: pass session_id from turn 1
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What did we just discuss?\", \"session_id\": \"<SESSION_ID>\"}"
```

### Example queries the agent handles well

| Query | Expected behavior |
|-------|-------------------|
| `What is LoRA fine-tuning?` | Local retrieve → cited answer |
| `Compare standard LoRA and layer-wise LoRA...` | Retrieve both sides → comparator |
| `Find recent papers on RLHF` | Fresh arXiv search |
| `Have you already read the LoRA paper before?` | Memory / papers_read check |
| `asdkjfh laksjdf compare papers???` | Refuse gibberish |
| `What embedding model does this system use?` | Refuse self-referential |

---

## Evaluation

27-question eval set in `eval/questions.json`, covering:

- `factual_rag`: single-paper Q&A with citations
- `multi_paper_synthesis`: cross-paper claims
- `compare_papers`: structured comparisons
- `agentic_routing`: search vs retrieve vs refuse
- `memory_recall`: multi-turn session memory
- `edge_case`: missing paper IDs, empty query

### Run eval

**Terminal 1: API pointed at a throwaway memory DB:**

```bash
# Windows PowerShell
$env:MEMORY_DB_PATH="data/eval_memory.db"
uvicorn api.main:api --port 8000

# macOS/Linux
MEMORY_DB_PATH=data/eval_memory.db uvicorn api.main:api --port 8000
```

**Terminal 2: eval client:**

```bash
# Fresh run (reset memory + regenerate answers + judge)
python eval/run_eval.py --reset-memory-db data/eval_memory.db --no-resume --out eval/results.csv

# Resume an interrupted generate phase
python eval/run_eval.py --out eval/results.csv

# Re-judge cached answers only (no /chat calls)
python eval/run_eval.py --rejudge eval/raw_responses.json --out eval/results.csv --judge-runs 3
```

Outputs:
- `eval/raw_responses.json`: checkpointed `/chat` answers (resumable)
- `eval/results.csv`: scores + judge reasoning
- `eval/METRICS.md`: before/after snapshot for resume claims

### Latest numbers (`results10.csv`)

| Category | n | reference_match | faithfulness | relevance | citation_precision | citation_recall | avg latency |
|----------|---|-----------------|--------------|-----------|--------------------|-----------------|-------------|
| factual_rag | 8 | 3.50 | 4.38 | 5.00 | 0.92 | 1.00 | 74s |
| multi_paper_synthesis | 3 | 3.00 | 4.00 | 5.00 | 0.50 | 0.33 | 101s |
| compare_papers | 4 | 3.33 | 4.33 | 5.00 | 0.67 | 0.67 | 135s |
| agentic_routing | 4 | 4.00 | 5.00 | 5.00 | 1.00 | 1.00 | 89s |
| memory_recall | 6 | 4.17 | 4.83 | 5.00 | 0.50 | 0.50 | 41s |
| edge_case | 2 | 5.00 | 5.00 | 5.00 | n/a | n/a | 46s |

Scores are 1 to 5 (LLM-as-judge, median of 3 runs). Citation metrics only apply when `expected_sources` is non-empty.

On gold-sourced content questions (factual + multi-paper + compare): **75% citation precision and 75% citation recall** (was about 50% / 55% on `results9`). Factual RAG alone is **92% precision / 100% recall**. Latency on non-trivial queries: p50 about 79s, p95 about 147s (local `llama3.1:8b`).

**Strengths:** factual citations, routing, memory recall behavior, refusals for gibberish and self-referential questions.

**Known gaps:** multi-paper and some compare queries still pull LoRA variants instead of every expected side (especially when the query does not name the second paper clearly). Answer completeness vs the reference is often mid-3s even when the right paper is cited.

---

## Design highlights

- **Prefer local first:** first hop retrieves from the ingested store unless the user asks for recent/new papers
- **Title-aware ranking:** prefer introducing titles (`Concept: ...`) over hyphenated variants (`Concept-Variant`) without hardcoding paper IDs
- **Comparison aspects:** named A-vs-B queries re-rank (and, if needed, do one targeted fetch) so each side is represented
- **Citation hygiene:** clamp out-of-range `[n]` markers; renumber sources to match brackets; put the best title match first as `[1]`
- **LLM deadlines:** streaming Ollama calls with wall-clock budgets and `keep_alive` so mid-eval stalls from model eviction are rarer
- **Per-node timing:** uvicorn logs `[timing] synthesizer: 95.3s` so latency spikes are attributable

---

## Tests

```bash
python -m pytest tests/test_smoke.py -q
python tests/week1_testing.py
python tests/week2_testing.py
python tests/week3_testing.py
```

Manual checklists live in `tests/week*_manual.md`.

---

## MCP (optional standalone)

The agent talks to arXiv search through `tools/arxiv_mcp_client.py`, which launches `mcp_server/arxiv_mcp_server.py` as a stdio subprocess.

To inspect the server alone:

```bash
python mcp_server/arxiv_mcp_server.py   # waits on stdio; Ctrl+C to exit
```

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `/chat` hangs or 500 | Is Ollama running? `ollama list` / `ollama serve` |
| Empty / weak retrieval | Ingest core papers; check `data/chroma/` exists and has chunks |
| Eval pollutes real memory | Always set `MEMORY_DB_PATH=data/eval_memory.db` for eval |
| Resume skips questions | Delete `eval/raw_responses.json` or pass `--no-resume` |
| Numpy / torch import errors | Reinstall `sentence-transformers` in a clean venv; embeddings load lazily on first retrieve |

---

## License

Personal / portfolio project. Paper content remains under each paper’s original license (typically arXiv terms).
