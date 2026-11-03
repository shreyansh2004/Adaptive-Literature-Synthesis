"""
eval/run_eval.py

Day 24 — run the eval set against the running FastAPI /chat endpoint and
score faithfulness, answer relevance, and citation accuracy.

PREREQUISITE — run the server against a throwaway memory DB so eval runs
are reproducible and don't pollute your real memory.db:

    # Windows (PowerShell)
    $env:MEMORY_DB_PATH="data/eval_memory.db"
    uvicorn api.main:api --port 8000

    # macOS/Linux
    MEMORY_DB_PATH=data/eval_memory.db uvicorn api.main:api --port 8000

Then delete data/eval_memory.db before each fresh eval run (this script
does that for you if you pass --reset-memory-db, but it can only delete
the file — you still need the server pointed at that path).

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --reset-memory-db data/eval_memory.db
    python eval/run_eval.py --questions eval/questions.json --out eval/results.csv
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import requests

API_URL = os.environ.get("EVAL_API_URL", "http://localhost:8000")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3.1:8b")

# How much of each answer the judge sees. The old value (900 chars) was
# cutting the longest/most important answers (compare_papers,
# multi_paper_synthesis — often 1500-2500+ chars) off mid-sentence,
# sometimes before the actual conclusion the question was testing for.
# Bump this up; almost all answers fit under ~3500 chars.
JUDGE_MAX_CHARS = int(os.environ.get("JUDGE_MAX_CHARS", "3500"))

# Run the judge multiple times per answer and take the median. A single
# call from a small local judge model isn't reliably deterministic even at
# temperature=0 — q01/q17/q20a (near-identical "What is LoRA fine-tuning?"
# answers) scored faithfulness 4/2/4 on a single-pass run. Repeating and
# taking the median smooths that out; the spread also flags which rows to
# sanity-check by hand (see judge_disagreement in the output).
#
# Default is 1 — matches the original latency. /chat calls (170-360s each
# in your last run) are the real bottleneck, not the judge, so this only
# multiplies cheaply once generation is decoupled from judging (see
# --rejudge below). Bump JUDGE_RUNS (env var or --judge-runs) when you're
# doing a rejudge-only pass over already-cached answers.
JUDGE_RUNS = int(os.environ.get("JUDGE_RUNS", "1"))

# Content questions: reference describes the factual claims a good answer must convey.
JUDGE_CONTENT_PROMPT = """You are a strict grader. Compare the system's answer to the REFERENCE answer.
Your job is to check whether the system conveys the SAME core claims as the reference —
not whether the system answer merely sounds plausible on its own.

Question: {query}

REFERENCE answer (the ground truth the system must match):
{reference_answer}

SYSTEM's actual answer:
{system_answer}

Returned source arXiv IDs (may be empty): {returned_ids}
Expected source arXiv IDs (may be empty): {expected_ids}
Tool calls made (may be empty): {tool_calls}

Score 1-5 on EACH dimension:

1) reference_match: Does the system answer convey the same key claims / conclusion as the REFERENCE?
   - 5 = covers the reference's core claims accurately (wording may differ)
   - 3 = partially overlaps the reference but misses important claims or adds unrelated ones
   - 1 = contradicts the reference, answers a different question, or ignores the reference claims
   IMPORTANT: Do NOT give a high score just because the answer is fluent or on-topic.
   It must align with the REFERENCE content.

2) faithfulness: Relative to the REFERENCE, does the system avoid fabricating claims the
   reference does not support?
   - 5 = no unsupported inventions beyond the reference
   - 1 = invents major claims absent from / contradicting the reference

3) relevance: Does the system answer the QUESTION asked?
   - 5 = directly answers the question
   - 1 = off-topic or non-answer

Respond ONLY with valid JSON:
{{"reference_match": <int 1-5>, "faithfulness": <int 1-5>, "relevance": <int 1-5>, "reasoning": "<one short sentence citing what matched or missed vs the reference>"}}
"""

# Behavioral questions: reference describes required agent *behavior* (routing, memory, refusal).
JUDGE_BEHAVIOR_PROMPT = """You are grading whether the system behaved as the REFERENCE instructs.
The reference describes the CORRECT behavior (search vs retrieve, refuse, recall memory, etc.),
not a factual paper claim. If the system did what the reference describes, that IS a 5 —
do not invent extra requirements the reference never states.

Question: {query}

REFERENCE behavior (this is the CORRECT behavior the system should show):
{reference_answer}

SYSTEM's actual answer:
{system_answer}

VERIFIED FACTS about what the system actually did (derived from execution logs —
these are ground truth; if your impression contradicts them, the FACTS win):
{action_facts}

Returned source arXiv IDs (may be empty): {returned_ids}
Expected source arXiv IDs (may be empty): {expected_ids}

Score 1-5 on EACH dimension:

1) reference_match: Does the system's answer/behavior match what the REFERENCE requires?
   - 5 = matches the required behavior (e.g. refused gibberish, recalled prior topic,
     searched for a new topic, said paper was already read, etc.)
   - 3 = partially matches
   - 1 = violates the required behavior (e.g. fabricated a comparison for nonsense input,
     ignored memory, searched when it should have answered locally)
   RULES:
   - Base routing judgments ONLY on the VERIFIED FACTS above, never on guesses.
   - If the reference says the system should refuse or say "cannot answer", then a refusal
     IS the correct behavior and scores 5.
   - Do not penalize the system for including helpful detail beyond the minimum, as long
     as the required behavior happened.

2) faithfulness: Did the system avoid fabricating content that the behavior rules forbid?
   - 5 = honest / no fabrication
   - 1 = fabricated papers, claims, or prior context it should not have

3) relevance: Does the response appropriately address the user's question/situation?
   - 5 = appropriate response
   - 1 = irrelevant

Respond ONLY with valid JSON:
{{"reference_match": <int 1-5>, "faithfulness": <int 1-5>, "relevance": <int 1-5>, "reasoning": "<one short sentence>"}}
"""

BEHAVIORAL_REFERENCE_PREFIXES = (
    "should ",
    "requires ",
    "in a brand-new",
    "this is a question about the system's",
)

# Categories whose reference answers describe required agent behavior.
# compare_papers / multi_paper_synthesis references may be phrased as "Should
# produce a comparison covering..." but they still describe ANSWER CONTENT, so
# they are graded with the content prompt (the behavior prompt caused the
# judge to invent routing requirements for them in results6).
BEHAVIORAL_CATEGORIES = {"agentic_routing", "edge_case"}


def _derive_action_facts(tool_calls: list) -> str:
    """
    Convert raw tool-call strings into plain yes/no statements the judge
    can't misread. In results6 the judge claimed q16 'should have searched'
    when tool_calls plainly showed planner:search_arxiv -> search_arxiv(5),
    and claimed q17 'triggered a new search' when it only did retrieve_local.
    Feeding pre-digested facts (instead of raw strings) prevents that.
    """
    calls = " | ".join(tool_calls or [])
    searched = "search_arxiv(" in calls
    retrieved = "retrieve_local(" in calls
    compared = "comparator(" in calls
    refused_gibberish = "gibberish" in calls
    self_ref = "self_referential" in calls
    from_memory = "from_memory" in calls or "paper_read_status" in calls
    not_found = "paper_not_found" in calls

    facts = [
        f"- Searched arXiv for new papers: {'YES' if searched else 'NO'}",
        f"- Retrieved from the local ingested store: {'YES' if retrieved else 'NO'}"
        + (" (NOTE: retrieval after a search is a normal part of the search pipeline —"
           " searched papers are ingested locally and then retrieved; do not treat this"
           " as 'relying on old local context')" if searched and retrieved else ""),
        f"- Ran the paper-comparison node: {'YES' if compared else 'NO'}",
        f"- Refused the query as unclear/gibberish: {'YES' if refused_gibberish else 'NO'}",
        f"- Declined because the question was about the system itself: {'YES' if self_ref else 'NO'}",
        f"- Answered from session/paper memory: {'YES' if from_memory else 'NO'}",
        f"- Reported a paper as not found: {'YES' if not_found else 'NO'}",
    ]
    return "\n".join(facts)


def smart_truncate(text: str, max_chars: int) -> str:
    """Truncate at the last sentence boundary before max_chars, rather than
    hard-cutting mid-sentence (which is what the old text[:900] slicing did).
    Falls back to a word boundary if no sentence break is found."""
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    for boundary in (". ", ".\n", "! ", "? "):
        cut = window.rfind(boundary)
        if cut != -1 and cut > max_chars * 0.5:  # don't cut too early
            return window[: cut + 1] + " [...truncated for judging...]"

    # No good sentence boundary found — fall back to a word boundary.
    cut = window.rfind(" ")
    if cut == -1:
        cut = max_chars
    return window[:cut] + " [...truncated for judging...]"


def is_behavioral_question(category: Optional[str], reference_answer: str) -> bool:
    """
    Behavioral refs describe what the agent should *do*, not what a correct
    answer should *say*. Classification is category-first: compare_papers and
    multi_paper_synthesis questions are always content-graded even when their
    reference happens to start with "Should produce..." — in results6 the
    prefix-only heuristic sent q11-q15 to the behavior judge, which then
    hallucinated routing complaints about them.
    """
    if category in BEHAVIORAL_CATEGORIES:
        return True
    if category == "memory_recall":
        # Turn-1 questions of a memory pair (q20a, q21a) have factual
        # references; the recall turns (q20b, q22, q23) have "Should ..." refs.
        ref_l = (reference_answer or "").strip().lower()
        return ref_l.startswith(BEHAVIORAL_REFERENCE_PREFIXES)
    if category == "factual_rag":
        # q08-style adversarial items describe expected refusal behavior.
        ref_l = (reference_answer or "").strip().lower()
        return ref_l.startswith("this is a question about the system's")
    return False


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_cache(path: str) -> dict:
    """id -> raw record (pre-judging). Empty dict if the file doesn't exist yet."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def save_raw_cache(path: str, cache: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(cache.values()), f, indent=2)


def reset_memory_db(path: str):
    p = Path(path)
    if p.exists():
        p.unlink()
        print(f"[reset] deleted {path}")
    else:
        print(f"[reset] {path} did not exist, nothing to delete")


def call_chat(query: str, session_id: Optional[str] = None) -> dict:
    payload = {"query": query}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=900)
    resp.raise_for_status()
    return resp.json()


def _judge_call_once(prompt: str) -> Optional[dict]:
    """Single judge call. Returns None on any failure so the caller can
    decide how to handle a partial batch of runs."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "options": {"temperature": 0},
            "stream": False,
        }, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        parsed = json.loads(raw)
        return {
            "reference_match": int(parsed.get("reference_match", parsed.get("faithfulness", 0))),
            "faithfulness": int(parsed.get("faithfulness", 0)),
            "relevance": int(parsed.get("relevance", 0)),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        print(f"  [judge] single run failed: {e}")
        return None


def judge_answer(
    query: str,
    reference_answer: str,
    system_answer: str,
    *,
    category: Optional[str] = None,
    returned_ids: Optional[list] = None,
    expected_ids: Optional[list] = None,
    tool_calls: Optional[list] = None,
) -> dict:
    """
    Grade the system answer against the reference answer.

    Uses a content prompt (must match reference claims) or a behavior prompt
    (must match required agent behavior), depending on the question type.
    """
    behavioral = is_behavioral_question(category, reference_answer)
    template = JUDGE_BEHAVIOR_PROMPT if behavioral else JUDGE_CONTENT_PROMPT

    format_args = {
        "query": query,
        "reference_answer": smart_truncate(reference_answer or "", JUDGE_MAX_CHARS),
        "system_answer": smart_truncate(system_answer or "", JUDGE_MAX_CHARS),
        "returned_ids": ", ".join(returned_ids or []) or "(none)",
        "expected_ids": ", ".join(expected_ids or []) or "(none)",
    }
    if behavioral:
        format_args["action_facts"] = _derive_action_facts(tool_calls or [])
    else:
        format_args["tool_calls"] = ", ".join(tool_calls or []) or "(none)"

    prompt = template.format(**format_args)

    runs = [r for r in (_judge_call_once(prompt) for _ in range(JUDGE_RUNS)) if r is not None]

    if not runs:
        return {
            "reference_match": 0, "faithfulness": 0, "relevance": 0,
            "faithfulness_spread": 0, "relevance_spread": 0,
            "judge_disagreement": False,
            "judge_reasoning": "judge error: all runs failed",
            "judge_mode": "behavior" if behavioral else "content",
        }

    match_scores = [r["reference_match"] for r in runs]
    faith_scores = [r["faithfulness"] for r in runs]
    rel_scores = [r["relevance"] for r in runs]
    match_median = statistics.median(match_scores)
    faith_median = statistics.median(faith_scores)
    rel_median = statistics.median(rel_scores)
    faith_spread = max(faith_scores) - min(faith_scores)
    rel_spread = max(rel_scores) - min(rel_scores)
    match_spread = max(match_scores) - min(match_scores)

    disagreement = faith_spread >= 2 or rel_spread >= 2 or match_spread >= 2

    reasoning = next(
        (r["reasoning"] for r in runs if r["reference_match"] == match_median),
        runs[0]["reasoning"],
    )
    if disagreement:
        reasoning += (
            f" [judge disagreement across {len(runs)} runs: "
            f"reference_match={match_scores}, faithfulness={faith_scores}, relevance={rel_scores}]"
        )

    return {
        "reference_match": int(match_median),
        "faithfulness": int(faith_median),
        "relevance": int(rel_median),
        "faithfulness_spread": faith_spread,
        "relevance_spread": rel_spread,
        "judge_disagreement": disagreement,
        "judge_reasoning": reasoning,
        "judge_mode": "behavior" if behavioral else "content",
    }


def score_citations(returned_sources: list[dict], expected_sources: list[str]) -> dict:
    if not expected_sources:
        # Nothing to check against — don't penalize, just mark N/A.
        return {"citation_precision": None, "citation_recall": None}

    returned_ids = {s.get("arxiv_id") for s in returned_sources if s.get("arxiv_id")}
    expected_ids = set(expected_sources)

    if not returned_ids:
        return {"citation_precision": 0.0, "citation_recall": 0.0}

    true_positives = returned_ids & expected_ids
    precision = len(true_positives) / len(returned_ids)
    recall = len(true_positives) / len(expected_ids)
    return {"citation_precision": round(precision, 2), "citation_recall": round(recall, 2)}


def generate(questions_path: str, raw_cache_path: str, reset_db_path: Optional[str],
             limit: Optional[int] = None, resume: bool = True):
    """Phase 1: call /chat for each question and cache the raw answer/sources.
    Checkpoints after every question, so a mid-run timeout only costs you
    the in-flight question, not the whole batch — rerun the same command
    and already-cached questions are skipped."""
    if reset_db_path:
        reset_memory_db(reset_db_path)

    try:
        requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"ERROR: API not reachable at {API_URL}: {e}")
        sys.exit(1)

    questions = load_questions(questions_path)
    if limit:
        questions = questions[:limit]
        print(f"[limit] running only the first {limit} question(s)")

    cache = load_raw_cache(raw_cache_path) if resume else {}
    if cache:
        print(f"[resume] {len(cache)} question(s) already cached in {raw_cache_path}, will skip those")

    session_ids: dict[str, str] = {qid: r.get("session_id") for qid, r in cache.items() if r.get("session_id")}
    run_start = time.time()

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["query"]

        if qid in cache:
            print(f"[skip {i}/{len(questions)}] {qid} — already cached")
            continue

        requires_session = q.get("requires_session", False)
        chain_from = q.get("chain_from")

        session_id = None
        if requires_session and chain_from:
            session_id = session_ids.get(chain_from)
            if session_id is None:
                print(f"[warn] {qid} requires_session but chain_from={chain_from!r} has no recorded session_id yet — running without one")

        print(f"[run {i}/{len(questions)}] {qid} ({q.get('category')}): {query!r}")

        if not query.strip():
            # Special case: q25, empty query should 400, not go through the judge.
            try:
                resp = requests.post(f"{API_URL}/chat", json={"query": query}, timeout=30)
                status = resp.status_code
            except Exception as e:
                status = f"error: {e}"
            cache[qid] = {
                "id": qid, "category": q.get("category"), "query": query,
                "reference_answer": q.get("reference_answer", ""),
                "expected_sources": q.get("expected_sources", []),
                "expected_status": 400, "actual_status": status,
                "answer": "", "sources": [], "tool_calls": [], "elapsed_sec": None,
                "is_empty_query_test": True,
            }
            save_raw_cache(raw_cache_path, cache)
            continue

        t0 = time.time()
        try:
            response = call_chat(query, session_id=session_id)
        except Exception as e:
            print(f"  [error] chat call failed: {e}")
            cache[qid] = {
                "id": qid, "category": q.get("category"), "query": query,
                "reference_answer": q.get("reference_answer", ""),
                "expected_sources": q.get("expected_sources", []),
                "expected_status": 200, "actual_status": f"error: {e}",
                "answer": "", "sources": [], "tool_calls": [], "elapsed_sec": None,
                "request_failed": True,
            }
            save_raw_cache(raw_cache_path, cache)
            continue
        elapsed = time.time() - t0

        session_ids[qid] = response.get("session_id")
        cache[qid] = {
            "id": qid,
            "category": q.get("category"),
            "query": query,
            "reference_answer": q.get("reference_answer", ""),
            "expected_sources": q.get("expected_sources", []),
            "expected_status": 200,
            "actual_status": 200,
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "tool_calls": response.get("tool_calls", []),
            "session_id": response.get("session_id"),
            "elapsed_sec": round(elapsed, 1),
        }
        # Checkpoint immediately — this is what makes the run resumable.
        save_raw_cache(raw_cache_path, cache)

        # Show what the agent actually did for this query (planner routing,
        # arXiv search, fetch/ingest cache hits, local retrieval, ...).
        # tool_calls = response.get("tool_calls", [])
        # if tool_calls:
        #     print(f"  steps: {' -> '.join(tool_calls)}")
        # n_sources = len(response.get("sources", []))
        # print(f"  sources: {n_sources}")

        total_elapsed = time.time() - run_start
        print(f"  done in {elapsed:.1f}s (total so far: {total_elapsed/60:.1f} min)")

    print(f"\n[done] generation phase cached {len(cache)} question(s) in {raw_cache_path}")
    return cache


def judge_phase(cache: dict, out_path: str):
    """Phase 2: judge whatever's in the raw cache. Never calls /chat, so this
    is safe (and fast) to re-run with different JUDGE_RUNS/JUDGE_MAX_CHARS
    values without regenerating anything."""
    results = []
    for qid, r in cache.items():
        if r.get("is_empty_query_test"):
            results.append({
                "id": r["id"], "category": r.get("category"), "query": r.get("query"),
                "expected_status": r.get("expected_status"), "actual_status": r.get("actual_status"),
                "answer": "", "reference_match": None, "faithfulness": None, "relevance": None,
                "judge_disagreement": False, "judge_mode": "",
                "citation_precision": None, "citation_recall": None,
                "judge_reasoning": "empty-query validation test, not judged",
            })
            continue

        if r.get("request_failed"):
            results.append({
                "id": r["id"], "category": r.get("category"), "query": r.get("query"),
                "expected_status": r.get("expected_status"), "actual_status": r.get("actual_status"),
                "answer": "", "reference_match": None, "faithfulness": None, "relevance": None,
                "judge_disagreement": False, "judge_mode": "",
                "citation_precision": None, "citation_recall": None,
                "judge_reasoning": "request failed",
            })
            continue

        returned_sources = r.get("sources", [])
        returned_ids = sorted({
            s.get("arxiv_id", "") for s in returned_sources if s.get("arxiv_id")
        })
        expected_ids = r.get("expected_sources", []) or []
        judge_scores = judge_answer(
            r["query"],
            r.get("reference_answer", ""),
            r.get("answer", ""),
            category=r.get("category"),
            returned_ids=returned_ids,
            expected_ids=expected_ids,
            tool_calls=r.get("tool_calls") or [],
        )
        citation_scores = score_citations(returned_sources, expected_ids)

        results.append({
            "id": r["id"],
            "category": r.get("category"),
            "query": r.get("query"),
            "expected_status": r.get("expected_status"),
            "actual_status": r.get("actual_status"),
            "answer": r.get("answer", ""),
            "elapsed_sec": r.get("elapsed_sec"),
            "reference_match": judge_scores["reference_match"],
            "faithfulness": judge_scores["faithfulness"],
            "relevance": judge_scores["relevance"],
            "judge_disagreement": judge_scores.get("judge_disagreement", False),
            "judge_mode": judge_scores.get("judge_mode", ""),
            "citation_precision": citation_scores["citation_precision"],
            "citation_recall": citation_scores["citation_recall"],
            "returned_arxiv_ids": ",".join(returned_ids),
            "expected_arxiv_ids": ",".join(expected_ids),
            "judge_reasoning": judge_scores["judge_reasoning"],
        })

    write_csv(results, out_path)
    print_summary(results)
    return results


def write_csv(results: list[dict], out_path: str):
    fieldnames = [
        "id", "category", "query", "expected_status", "actual_status",
        "answer", "elapsed_sec", "reference_match", "faithfulness", "relevance",
        "judge_disagreement", "judge_mode",
        "citation_precision", "citation_recall",
        "returned_arxiv_ids", "expected_arxiv_ids", "judge_reasoning",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\n[done] wrote {len(results)} rows to {out_path}")


def print_summary(results: list[dict]):
    print("\n=== Summary by category ===")
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    for cat, rows in by_cat.items():
        match = avg([r.get("reference_match") for r in rows])
        faith = avg([r["faithfulness"] for r in rows])
        rel = avg([r["relevance"] for r in rows])
        prec = avg([r["citation_precision"] for r in rows])
        rec = avg([r["citation_recall"] for r in rows])
        print(f"{cat:22s} n={len(rows):2d}  reference_match={match}  "
              f"faithfulness={faith}  relevance={rel}  "
              f"citation_precision={prec}  citation_recall={rec}")

    all_match = avg([r.get("reference_match") for r in results])
    all_faith = avg([r["faithfulness"] for r in results])
    all_rel = avg([r["relevance"] for r in results])
    print(f"\n{'OVERALL':22s} n={len(results):2d}  reference_match={all_match}  "
          f"faithfulness={all_faith}  relevance={all_rel}")

    flagged = [r for r in results if r.get("judge_disagreement")]
    if flagged:
        print(f"\n=== {len(flagged)} row(s) with judge disagreement across runs (spot-check these) ===")
        for r in flagged:
            print(f"  {r['id']}: {r['query']!r}")

    low_match = [
        r for r in results
        if r.get("reference_match") is not None and r["reference_match"] <= 2
    ]
    if low_match:
        print(f"\n=== {len(low_match)} row(s) with reference_match <= 2 ===")
        for r in low_match:
            print(f"  {r['id']}: {r.get('judge_reasoning', '')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--out", default="eval/results.csv")
    parser.add_argument("--raw-cache", default="eval/raw_responses.json",
                         help="Where raw /chat answers are checkpointed before judging. "
                              "Reused automatically on rerun so an interrupted run resumes "
                              "instead of restarting from scratch.")
    parser.add_argument("--reset-memory-db", default=None,
                         help="Path to the eval memory DB file to delete before running "
                              "(must match MEMORY_DB_PATH the server was launched with)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N questions — useful for a quick smoke "
                              "test before committing to a full run")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing --raw-cache and regenerate every answer "
                              "from scratch instead of skipping already-cached questions.")
    parser.add_argument("--rejudge", metavar="CACHE_PATH", default=None,
                         help="Skip generation entirely and just (re-)judge answers already "
                              "saved in this raw-cache JSON file. No /chat calls are made, so "
                              "this is safe to run with a higher --judge-runs without risking "
                              "a timeout on the slow agent pipeline.")
    parser.add_argument("--judge-runs", type=int, default=None,
                         help=f"Override JUDGE_RUNS (default {JUDGE_RUNS}, or $JUDGE_RUNS). "
                              "Only worth raising when paired with --rejudge, since each extra "
                              "run re-queries the (cheap, local) judge model, not /chat.")
    args = parser.parse_args()

    if args.judge_runs is not None:
        JUDGE_RUNS = args.judge_runs

    if args.rejudge:
        cache = load_raw_cache(args.rejudge)
        if not cache:
            print(f"ERROR: no cached answers found at {args.rejudge}")
            sys.exit(1)
        print(f"[rejudge] scoring {len(cache)} cached answer(s) from {args.rejudge}, "
              f"JUDGE_RUNS={JUDGE_RUNS} (no /chat calls will be made)")
        judge_phase(cache, args.out)
    else:
        cache = generate(args.questions, args.raw_cache, args.reset_memory_db,
                          limit=args.limit, resume=not args.no_resume)
        judge_phase(cache, args.out)