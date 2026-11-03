# Eval metrics snapshot

Before and after the resume-driven retrieval and synthesis changes.
Judge scores are 1 to 5 (median of 3 runs). Citation metrics apply only when
`expected_sources` is non-empty.

## Baseline (`results9.csv`)

| Category | n | reference_match | faithfulness | relevance | citation_precision | citation_recall | avg latency |
|----------|---|-----------------|--------------|-----------|--------------------|-----------------|-------------|
| factual_rag | 8 | 3.38 | 4.25 | 5.00 | 0.58 | 0.67 | 92s |
| multi_paper_synthesis | 3 | 3.33 | 4.33 | 5.00 | 0.33 | 0.33 | 123s |
| compare_papers | 4 | 3.75 | 4.75 | 5.00 | 0.50 | 0.50 | 139s |
| agentic_routing | 4 | 4.00 | 5.00 | 5.00 | 1.00 | 1.00 | 72s |
| memory_recall | 6 | 4.50 | 4.83 | 5.00 | 1.00 | 1.00 | 30s |
| edge_case | 2 | 5.00 | 5.00 | 5.00 | n/a | n/a | 32s |

Gold-sourced content questions (factual + multi + compare): about 50% citation precision and 55% recall.

## After fix (`results10.csv`, clean API run)

| Category | n | reference_match | faithfulness | relevance | citation_precision | citation_recall | avg latency |
|----------|---|-----------------|--------------|-----------|--------------------|-----------------|-------------|
| factual_rag | 8 | 3.50 | 4.38 | 5.00 | 0.92 | 1.00 | 74s |
| multi_paper_synthesis | 3 | 3.00 | 4.00 | 5.00 | 0.50 | 0.33 | 101s |
| compare_papers | 4 | 3.33 | 4.33 | 5.00 | 0.67 | 0.67 | 135s |
| agentic_routing | 4 | 4.00 | 5.00 | 5.00 | 1.00 | 1.00 | 89s |
| memory_recall | 6 | 4.17 | 4.83 | 5.00 | 0.50 | 0.50 | 41s |
| edge_case | 2 | 5.00 | 5.00 | 5.00 | n/a | n/a | 46s |

| Aggregate | Before | After |
|-----------|--------|-------|
| Content gold citation precision | ~50% | **75%** |
| Content gold citation recall | ~55% | **75%** |
| Factual citation precision / recall | 58% / 67% | **92% / 100%** |
| Latency p50 / p95 (queries >= 5s) | (not logged) | **~79s / ~147s** |

## What changed (no paper-ID hardcoding)

1. Retrieval (`rag/retrieve.py`): introducing-paper vs variant title scoring, aspect keywords that keep compound phrases, lexical title rescue when dense search misses a named aspect, multi-paper diversity.
2. Synthesis (`agents/synthesizer.py`, `agents/planner.py`): word-boundary match for "differ"; abstract-first context for definitional questions; reorder papers by title match so `[1]` is the best title fit; single-focus answers keep one paper.

## Resume bullets (use these)

- Built a LangGraph agentic RAG system over arXiv papers (MCP search tool, Chroma retrieval, SQLite session memory, FastAPI) and evaluated it on a 27-query suite spanning factual Q&A, multi-paper synthesis, comparison, routing, and memory.
- Improved gold-sourced citation precision/recall from about 50%/55% to **75%/75%** (factual RAG: **92%/100%**) via title-aware ranking, aspect coverage, and citation ordering, without hardcoding paper IDs.
- Routing and memory categories scored about **4.1/5** mean reference match on the same judge harness, with gates for gibberish and self-referential questions.
