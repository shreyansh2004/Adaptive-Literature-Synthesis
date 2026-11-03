# Agentic ML/AI Research Assistant — 4-Week Build Plan

**Goal:** A working, deployed, evaluated agentic research assistant by end of Week 4 (~end of July).
**Pace:** 10–20 hrs/week → roughly 1.5–2.5 hrs/day, 6 days/week. Sunday is buffer/rest.
**Principle:** Every day should end with something *runnable*, even if small. Don't let any single day become "just reading docs."

---

## Week 1 — Foundations: Data + Basic RAG

**Milestone by end of week:** You can ask a question and get a RAG-based answer with citations, over a small local set of papers.

- **Day 1 (Mon):** Repo setup. Create folder structure below. Set up virtualenv, install core libs (`langgraph`, `langchain`, `chromadb` or `faiss-cpu`, `fastapi`, `uvicorn`, `requests`). Get a "hello world" FastAPI endpoint running.
- **Day 2 (Tue):** Build `tools/arxiv_search.py` — standalone Python function that queries the arxiv API given a topic string, returns title/abstract/id/url for top N results. Test it in isolation (no agent yet).
- **Day 3 (Wed):** Build `tools/paper_fetch.py` — given an arxiv ID, fetch full text or abstract+metadata. Store raw fetched papers to disk (`data/raw/`) as JSON.
- **Day 4 (Thu):** Build ingestion pipeline: chunk paper text → embed (OpenAI/local embedding model) → store in Chroma/FAISS (`rag/ingest.py`, `rag/store.py`). Ingest ~10 papers on a topic you like (e.g., "LoRA fine-tuning" or "RAG evaluation").
- **Day 5 (Fri):** Build `rag/retrieve.py` — given a query, retrieve top-k chunks. Wire up a bare-bones "answer with citation" function (no agent framework yet, just a linear script: retrieve → prompt LLM → return answer + sources).
- **Day 6 (Sat):** Test end-to-end manually with 5-10 real questions. Fix retrieval quality issues (chunk size, overlap, embedding model choice). Write down which queries fail — you'll need these for Week 4 eval anyway.
- **Day 7 (Sun):** Buffer/rest. Optional: clean up code, write a short internal README note on what's done.

---

## Week 2 — Agentic Layer (LangGraph)

**Milestone by end of week:** The system decides *for itself* whether to search, retrieve, or answer directly, and can chain multiple steps.

- **Day 8 (Mon):** Learn LangGraph basics (state, nodes, edges, conditional routing) — just enough to build a 2-node graph. Build the `AgentState` schema (query, retrieved_docs, tool_calls, answer, etc).
- **Day 9 (Tue):** Build the **planner node**: given a query + current state, decide next action (search_arxiv / retrieve_local / answer_now). Use structured output (JSON) from the LLM for the decision.
- **Day 10 (Wed):** Wire the arxiv search and RAG retrieve functions from Week 1 as **tool nodes** in the graph. Get a simple 2-hop flow working: planner → search → planner → answer.
- **Day 11 (Thu):** Add the **synthesizer node**: combines retrieved chunks + search results into a final cited answer. Handle the case of multiple sources needing reconciliation (e.g., conflicting claims across papers).
- **Day 12 (Fri):** Add a "compare papers" capability — given 2+ paper IDs already retrieved, produce a structured comparison (method, results, limitations). This is a good differentiator for demo purposes.
- **Day 13 (Sat):** Stress-test with multi-step queries ("find recent papers on X, then compare the top 2 approaches"). Debug routing logic — this is usually the messiest part, budget extra time.
- **Day 14 (Sun):** Buffer/rest.

---

## Week 3 — Memory + MCP

**Milestone by end of week:** The agent remembers past sessions, and at least one tool is served via a real MCP server.

- **Day 15 (Mon):** Design memory schema: `papers_read`, `topics_covered`, `key_conclusions`, `user_preferences`. Decide storage (simple JSON/SQLite is fine — don't overbuild).
- **Day 16 (Tue):** Implement memory write: after each session, summarize what was covered and persist it. Implement memory read: inject relevant memory into the planner's context at session start.
- **Day 17 (Wed):** Add memory-aware behavior: if a paper was already read before, the agent should say so and use cached knowledge instead of re-fetching. Test across 2+ separate "sessions."
- **Day 18 (Thu):** Learn MCP basics (server/client model, tool exposure spec). Set up a minimal MCP server skeleton.
- **Day 19 (Fri):** Convert `arxiv_search` (or `paper_fetch`) into a proper MCP server tool. Get your LangGraph agent calling it as an MCP client instead of a direct function call.
- **Day 20 (Sat):** Test the full MCP-integrated flow end-to-end. Fix any protocol/connection issues (these tend to be fiddly — budget time here).
- **Day 21 (Sun):** Buffer/rest.

---

## Week 4 — Eval + Deploy + Polish

**Milestone by end of week:** Deployed, evaluated, demo-ready project with README and video.

- **Day 22 (Mon):** Wrap the LangGraph app in FastAPI (`/chat` endpoint, session ID handling for memory). Test via curl/Postman.
- **Day 23 (Tue):** Build eval set: 20-25 questions with reference answers (use the failure cases you noted back in Week 1, Day 6, plus new ones covering agentic + memory behavior).
- **Day 24 (Wed):** Run eval using RAGAS or a simple LLM-as-judge script — score faithfulness, answer relevance, citation accuracy. Log results to a table/CSV.
- **Day 25 (Thu):** Fix the worst-scoring failure modes from eval (this is what makes your project defensible in interviews — "I found X was weak, here's how I fixed it").
- **Day 26 (Fri):** Deploy (Render/Railway/HF Spaces). Handle env vars, API keys, cold-start issues.
- **Day 27 (Sat):** Write README (architecture diagram, setup instructions, eval numbers table, example queries + outputs). Record 60-90 sec demo video/GIF.
- **Day 28 (Sun):** Final review. Push to GitHub, pin it, write resume bullet points based on concrete numbers (e.g., "Built agentic RAG system achieving 89% citation accuracy across 25-query eval set; integrated MCP-based tool serving and persistent cross-session memory").

---

## Suggested Repo Structure

```
research-assistant/
├── agents/
│   ├── graph.py          # LangGraph state machine definition
│   ├── planner.py        # planner node logic
│   └── synthesizer.py    # answer synthesis node
├── tools/
│   ├── arxiv_search.py
│   └── paper_fetch.py
├── mcp_server/
│   └── arxiv_mcp_server.py
├── rag/
│   ├── ingest.py
│   ├── store.py
│   └── retrieve.py
├── memory/
│   ├── schema.py
│   └── store.py
├── api/
│   └── main.py            # FastAPI app
├── eval/
│   ├── questions.json     # eval question set + reference answers
│   └── run_eval.py
├── data/
│   └── raw/                # fetched paper JSON
├── README.md
└── requirements.txt
```

## Weekly Checkpoints (self-review questions)
- **End of Week 1:** Can I ask a factual question and get a correctly cited answer from local papers?
- **End of Week 2:** Does the agent correctly decide *when* to search vs. answer from memory/context, without me hardcoding the path?
- **End of Week 3:** If I ask about a paper I discussed yesterday, does it remember? Is at least one tool call going through MCP, not a direct function call?
- **End of Week 4:** Can a stranger clone my repo, read the README, and understand what this does and why it's good in under 2 minutes?

## Notes
- If a day's task overruns, don't skip the next day's task to catch up — instead trim scope (e.g., "compare papers" feature can slip to a stretch goal if Week 2 runs long). Protect the eval + deploy work in Week 4; that's what makes this resume-ready rather than just "a repo that exists."
- Keep a running note of design decisions and trade-offs as you go (why Chroma over FAISS, why this chunk size, etc.) — this becomes free interview prep material.
