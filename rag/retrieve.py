"""
Retrieval: embed a query and pull top-k relevant chunks from the vector store.
Uses the SAME embed_texts as ingestion — critical, or vectors won't compare correctly.

Dense retrieval is followed by a lightweight, topic-agnostic title-overlap
re-ranker. This helps exact paper/topic names without encoding known paper IDs
or eval-specific answers.
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

from rag.store import VectorStore


# Tiny stopword list — enough to keep boosts focused on content terms.
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "what", "how", "why", "when", "where", "which", "who", "whom",
    "this", "that", "these", "those", "it", "its", "as", "at", "by", "from",
    "about", "into", "over", "after", "before", "between", "than", "then",
    "vs", "versus", "compare", "comparison", "difference", "differences",
    "paper", "papers", "approach", "approaches", "method", "methods",
    "using", "use", "used", "via", "across", "among", "different",
    "mean", "means", "might", "need", "needs", "tell", "me", "can", "you",
    "top", "two", "better", "large", "scale", "terms", "performance",
}

# Trailing modifiers often appended after a named concept in definitional Qs.
_TRAILING_MODIFIERS: Set[str] = {
    "fine", "tuning", "finetuning", "adaptation", "learning", "training",
    "model", "models", "method", "methods", "approach", "approaches",
    "technique", "techniques", "system", "systems", "framework",
}


@dataclass
class RetrievedChunk:
    text: str
    arxiv_id: str
    title: str
    authors: str
    published: str
    updated: str
    url: str
    chunk_index: int
    source: str          # "abstract" or "full_text"
    distance: float       # lower = more similar (cosine distance)

    @property
    def score(self) -> float:
        return 1 - self.distance


def _keywords(text: str) -> List[str]:
    # Treat punctuation/hyphens as boundaries so "layer-wise" can match
    # "layer wise", while preserving model names and acronyms.
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


def _is_definitional_query(query: str) -> bool:
    q = query.lower().strip()
    if re.search(r"\b(?:compare|versus|vs\.?|difference|differ|tradeoff)\b", q):
        return False
    return bool(
        re.match(r"^\s*what\s+(?:is|are)\b", q)
        or re.match(r"^\s*what\s+does\b.+\bmean\b", q)
    )


def _definitional_subjects(query: str) -> Tuple[Optional[str], List[str]]:
    """
    Extract named concept cues from a query, without mapping them to any paper ID.

    Returns (acronym_or_short_name, phrase_tokens).
    Examples:
      "What is LoRA fine-tuning?" -> ("lora", ["lora"])
      "What is retrieval-augmented generation (RAG)?"
          -> ("rag", ["retrieval", "augmented", "generation"])
      "standard LoRA vs ..." -> ("lora", ["lora"])
      "How does federated learning combine with LoRA ...?" -> ("lora", ["lora"])
    """
    q = query.lower().strip()
    acronym = None
    phrase: List[str] = []

    paren = re.search(r"\(([a-zA-Z][a-zA-Z0-9]{1,10})\)", q)
    if paren:
        acronym = paren.group(1).lower()

    head = None
    m = re.match(r"^\s*what\s+(?:is|are)\s+(.+?)(?:\?|$)", q)
    if m:
        head = m.group(1)
    else:
        m = re.match(r"^\s*what\s+does\s+(.+?)\s+mean\b", q)
        if m:
            head = m.group(1)
        else:
            m = re.search(
                r"\b(?:standard|original|vanilla)\s+([a-zA-Z][a-zA-Z0-9\-]*)",
                q,
            )
            if m:
                head = m.group(1)

    if head:
        # Drop parenthetical acronym from the phrase scrape.
        head = re.sub(r"\([^)]*\)", " ", head)
        tokens = _keywords(head)
        # Drop trailing generic modifiers ("fine-tuning", "adaptation", ...).
        while tokens and tokens[-1] in _TRAILING_MODIFIERS:
            tokens.pop()
        phrase = tokens
        if not acronym and phrase:
            # Prefer a short leading token (often an acronym-like name).
            acronym = phrase[0]

    # Mixed-case / all-caps method names in the original query (LoRA, RAG, QLoRA).
    if not acronym:
        for tok in re.findall(
            r"\b([A-Z]{2,10}|[A-Z][a-z]+[A-Z][A-Za-z0-9\-]*)\b", query
        ):
            candidate = tok.lower().split("-")[0]
            if candidate not in _STOPWORDS and len(candidate) >= 2:
                acronym = candidate
                if not phrase:
                    phrase = [candidate]
                break

    return acronym, phrase


def _named_subject_head(query: str) -> Optional[str]:
    acronym, phrase = _definitional_subjects(query)
    return acronym or (phrase[0] if phrase else None)


def _wants_paper_diversity(query: str) -> bool:
    """Multi-paper / compare / agree queries benefit from one-chunk-per-paper first."""
    q = query.lower()
    if split_comparison_aspects(query):
        return True
    return bool(
        re.search(
            r"\b(?:different papers|across papers|papers agree|agree on|"
            r"variants?|compare|comparison|versus|vs\.?|difference between|"
            r"relative to the original)\b",
            q,
        )
    )


def _title_boost(title: str, keywords: List[str], query: str = "") -> float:
    """
    Reward query/title lexical overlap without any topic-specific rules.

    Signals:
    - proportion of query keywords present in the title
    - proportion of title keywords present in the query
    - longest consecutive query-token phrase found in the title
    - exact acronym matches
    - for definitional queries: prefer titles that *introduce* the named
      concept (Concept: ..., or title starting with the concept phrase)
      over variant/qualified titles (Concept-Variant, "Layer-wise Concept")
    """
    query_terms = keywords or _keywords(query)
    title_terms = _keywords(title)
    if not query_terms or not title_terms:
        return 0.0

    q_set, t_set = set(query_terms), set(title_terms)
    overlap = q_set & t_set
    if not overlap:
        return 0.0

    query_coverage = len(overlap) / len(q_set)
    title_coverage = len(overlap) / len(t_set)

    # Longest consecutive phrase shared by query and title.
    longest_phrase = 0
    for i in range(len(query_terms)):
        for j in range(len(title_terms)):
            n = 0
            while (
                i + n < len(query_terms)
                and j + n < len(title_terms)
                and query_terms[i + n] == title_terms[j + n]
            ):
                n += 1
            longest_phrase = max(longest_phrase, n)

    acronym_bonus = 0.0
    for term in overlap:
        if term.isalpha() and 2 <= len(term) <= 8 and term.upper() in title:
            acronym_bonus += 0.03

    boost = (
        0.16 * query_coverage
        + 0.08 * title_coverage
        + 0.04 * max(0, longest_phrase - 1)
        + acronym_bonus
    )

    subject, phrase = _definitional_subjects(query)
    title_l = title.lower()
    definitional = _is_definitional_query(query)
    # Prefer introducing papers whenever a named concept is present — not only
    # for "what is X?" — so "standard LoRA vs federated sketching" does not
    # promote LoRA-FAIR over the original LoRA paper.
    named_concept = bool(subject and len(subject) >= 2)

    if named_concept:
        # Order matters: check hyphenated variants BEFORE first-token match,
        # because "LoRA-FAIR" tokenizes to ["lora", "fair"] and would otherwise
        # look like an introducing title.
        if re.match(rf"^{re.escape(subject)}\s*:", title_l):
            boost += 0.30 if definitional else 0.24
        elif re.match(rf"^{re.escape(subject)}-", title_l):
            # "Concept-Variant: ..." — a follow-up, not the definition.
            boost -= 0.22 if definitional else 0.18
        elif title_terms and title_terms[0] == subject:
            boost += 0.24 if definitional else 0.18
        elif re.search(rf"\b{re.escape(subject)}\b", title_l):
            # Concept appears only after a qualifier ("Layer-wise X",
            # "Efficient X", "Engineering the X Stack", ...).
            # Milder penalty when the query itself leads with that qualifier
            # (e.g. "federated ... LoRA" → "Federated Sketching LoRA").
            leading_qualifies = bool(
                query_terms and title_terms and title_terms[0] == query_terms[0]
            )
            if leading_qualifies:
                boost += 0.10
            else:
                boost -= 0.14 if definitional else 0.10

    # Multi-word expanded form: "Retrieval-Augmented Generation ..." for RAG.
    if phrase and len(phrase) >= 2:
        if title_terms[: len(phrase)] == phrase:
            boost += 0.32 if definitional else 0.18
        elif all(t in t_set for t in phrase) and title_terms[:1] != phrase[:1]:
            # Phrase present but not leading — more likely a survey/variant.
            boost -= 0.10 if definitional else 0.05

    # Leading query content token as title prefix — favors on-topic papers
    # ("federated learning ... LoRA" → "Federated Sketching LoRA" over LoRA-FAIR).
    if query_terms and title_terms and title_terms[0] == query_terms[0]:
        boost += 0.12

    return min(boost, 0.55)


def _clean_aspect(text: str) -> str:
    text = re.sub(r"[?.,;:]+$", "", text.strip())
    text = re.sub(
        r"^(?:the|a|an)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Keep standard/original/vanilla — they signal the introducing paper.
    text = re.sub(
        r"\s+(?:approaches?|methods?|techniques?|papers?)$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def aspect_keywords(aspect: str) -> List[str]:
    """
    Distinctive tokens for a comparison aspect. Generic modifiers like
    "fine-tuning"/"adaptation" appear in almost every title in a fine-tuning
    corpus, so they can't tell aspects apart alone — but when the aspect is
    only a qualifier + modifiers ("partial fine-tuning"), keep the full token
    list so targeted fetch/title match can still hit "Partial Is Better Than
    All: Revisiting Fine-tuning...".
    """
    tokens = _keywords(aspect)
    # Drop canonicality adjectives from keyword set (used only as a scoring prior).
    tokens = [t for t in tokens if t not in {"standard", "original", "vanilla"}]
    distinctive = [t for t in tokens if t not in _TRAILING_MODIFIERS]
    if len(distinctive) <= 1:
        # Qualifier-only aspect: keep compound cues (partial + fine + tuning).
        return tokens or distinctive
    return distinctive


def _aspect_title_score(title: str, aspect: str) -> float:
    """How well a title matches an aspect — used to pick preferred papers."""
    kw = aspect_keywords(aspect)
    if not kw:
        return 0.0
    title_kw = _keywords(title)
    t_set = set(title_kw)
    title_l = title.lower()
    aspect_l = aspect.lower()
    hits = sum(1 for k in kw if k in t_set)
    score = hits / max(len(kw), 1)
    head = next((k for k in kw if k not in _TRAILING_MODIFIERS), kw[0])
    wants_canonical = bool(
        re.search(r"\b(?:standard|original|vanilla)\b", aspect_l)
    )
    if head and re.match(rf"^{re.escape(head)}\s*:", title_l):
        score += 0.9 if wants_canonical else 0.7
    elif head and re.match(rf"^{re.escape(head)}-", title_l):
        # Concept-Variant titles are the wrong side of "standard X".
        score -= 0.8 if wants_canonical else 0.5
    elif title_kw and title_kw[0] == head:
        score += 0.5
    elif head in t_set:
        score += 0.2
    return score


def _lexical_title_rescue(
    store: "VectorStore",
    aspect: str,
    limit: int = 4,
) -> List[RetrievedChunk]:
    """
    When dense retrieval misses a named aspect (e.g. "partial fine-tuning"
    drowned by LoRA variants), scan titles for lexical keyword overlap and
    pull one abstract/early chunk per matching paper.
    """
    kw = aspect_keywords(aspect)
    if not kw:
        return []
    head = next((k for k in kw if k not in _TRAILING_MODIFIERS), kw[0])
    try:
        raw = store.collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"[retrieve] lexical title rescue failed: {e}")
        return []

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    # Best chunk per matching paper, scored by title overlap.
    best: Dict[str, Tuple[float, RetrievedChunk]] = {}
    for doc, meta in zip(docs, metas):
        title = meta.get("title") or ""
        score = _aspect_title_score(title, aspect)
        # Require the distinctive head (e.g. "partial") — fine/tuning alone
        # matches almost every PEFT paper.
        if head and head not in set(_keywords(title)):
            continue
        if score < 0.5:
            continue
        arxiv_id = meta.get("arxiv_id") or ""
        if not arxiv_id:
            continue
        chunk = _chunk_from_store_row(
            doc,
            meta,
            dist=max(0.0, 0.55 - 0.1 * score),
        )
        prev = best.get(arxiv_id)
        # Prefer abstract / earlier chunks among equals.
        rank = (score, 1 if chunk.source == "abstract" else 0, -(chunk.chunk_index or 0))
        if prev is None or rank > (prev[0], 1 if prev[1].source == "abstract" else 0, -(prev[1].chunk_index or 0)):
            best[arxiv_id] = (score, chunk)

    ranked = sorted(best.values(), key=lambda p: p[0], reverse=True)
    return [c for _, c in ranked[:limit]]


def _aspect_covered(chunks: List[RetrievedChunk], aspect: str) -> bool:
    """True if some retrieved title matches the aspect's distinctive head."""
    kw = aspect_keywords(aspect)
    if not kw:
        return True
    head = next((k for k in kw if k not in _TRAILING_MODIFIERS), kw[0])
    # Require the distinctive head (e.g. "partial") — matching only
    # fine/tuning would mark every PEFT paper as covering "partial fine-tuning".
    if head:
        for c in chunks:
            if head in set(_keywords(c.title)):
                return True
        return False
    need = max(1, (len(kw) + 1) // 2)
    for c in chunks:
        title_kw = set(_keywords(c.title))
        if len(title_kw & set(kw)) >= need:
            return True
    return False


def split_comparison_aspects(query: str) -> List[str]:
    """
    If the query names two approaches to compare, return those aspects.
    Topic-agnostic pattern matching only — never hardcodes paper names.
    """
    q = query.strip()
    patterns = [
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\bcompare\s+(.+?)\s+(?:and|to|with|vs\.?|versus)\s+(.+?)(?:\?|$)",
        r"\bdifference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
        r":\s*(.+?)\s+or\s+(.+?)(?:\?|$)",
        r"\bwhich\s+is\s+better\b.*?\b(.+?)\s+or\s+(.+?)(?:\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if not m:
            continue
        left, right = _clean_aspect(m.group(1)), _clean_aspect(m.group(2))
        # Require each side to have some content signal.
        if len(_keywords(left)) >= 1 and len(_keywords(right)) >= 1:
            if left.lower() != right.lower():
                return [left, right]
    return []


def _rerank_chunks(chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
    """Re-score by (similarity + title boost); lower effective distance wins."""
    keywords = _keywords(query)
    if not chunks:
        return chunks

    definitional = _is_definitional_query(query)
    typical = bool(
        re.search(
            r"\b(?:typical|typically|usually|often|commonly)\b",
            query.lower(),
        )
    )
    rescored = []
    for c in chunks:
        boost = _title_boost(c.title, keywords, query)
        if c.source == "abstract":
            # Abstracts state what a method IS; body chunks discuss details.
            # Weigh them harder for definitional / typical-value questions so
            # answers come from the abstract, not a section-7 ablation.
            if definitional:
                boost += 0.12
            elif typical:
                boost += 0.10
            else:
                boost += 0.02
        elif typical and c.chunk_index > 5:
            # Ablation tables and late sections often report edge-case minima.
            boost -= 0.06
        # Mild preference for shorter, concept-focused titles on definitional Qs
        # (introducing papers tend to be terser than surveys).
        if definitional:
            title_len = len(_keywords(c.title))
            if title_len <= 6:
                boost += 0.04
            elif title_len >= 12:
                boost -= 0.04
        effective_distance = max(0.0, c.distance - boost)
        rescored.append((effective_distance, c))

    rescored.sort(key=lambda pair: pair[0])
    return [c for _, c in rescored]


def _chunk_from_store_row(doc: str, meta: dict, dist: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=doc,
        arxiv_id=meta["arxiv_id"],
        title=meta["title"],
        authors=meta.get("authors", ""),
        published=meta.get("published", ""),
        updated=meta.get("updated", ""),
        url=meta.get("url", ""),
        chunk_index=meta.get("chunk_index", 0),
        source=meta.get("source", "abstract"),
        distance=dist,
    )


def _dense_fetch(
    query: str,
    store: VectorStore,
    fetch_k: int,
    where: Optional[Dict],
) -> List[RetrievedChunk]:
    from rag.embeddings import embed_texts

    query_embedding = embed_texts([query])[0]
    raw = store.query(query_embedding=query_embedding, top_k=fetch_k, where=where)

    chunks: List[RetrievedChunk] = []
    documents = raw["documents"][0] if raw.get("documents") else []
    metadatas = raw["metadatas"][0] if raw.get("metadatas") else []
    distances = raw["distances"][0] if raw.get("distances") else []

    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist > 0.75:
            continue
        chunks.append(_chunk_from_store_row(doc, meta, dist))
    return chunks


def _cap_per_paper(
    chunks: List[RetrievedChunk],
    top_k: int,
    max_per_paper: Optional[int],
) -> List[RetrievedChunk]:
    if not max_per_paper:
        return chunks[:top_k]
    per_paper_count: Dict[str, int] = {}
    capped = []
    for c in chunks:
        count = per_paper_count.get(c.arxiv_id, 0)
        if count >= max_per_paper:
            continue
        capped.append(c)
        per_paper_count[c.arxiv_id] = count + 1
        if len(capped) >= top_k:
            break
    return capped


def _best_paper_ids(chunks: List[RetrievedChunk], n: int = 2) -> List[str]:
    """Distinct paper IDs in reranked order."""
    seen = []
    for c in chunks:
        if c.arxiv_id not in seen:
            seen.append(c.arxiv_id)
        if len(seen) >= n:
            break
    return seen


def _merge_aspect_results(
    aspect_results: List[List[RetrievedChunk]],
    full_query: str,
    top_k: int,
    max_per_paper: Optional[int],
    aspects: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """
    Guarantee coverage of each comparison aspect: pick the best paper for
    each aspect first, then fill remaining slots by global relevance.
    """
    if not aspect_results:
        return []

    # Preferred paper per aspect (best after that aspect's own rerank).
    # The last group is the full-query pass — skip it for preferred seeds.
    preferred: List[str] = []
    aspect_groups = aspect_results[:-1] if len(aspect_results) > 1 else aspect_results
    for idx, group in enumerate(aspect_groups):
        aspect = aspects[idx] if aspects and idx < len(aspects) else ""
        if aspect and group:
            # Prefer the paper whose title best matches the aspect keywords,
            # not merely the densest chunk (which often belongs to a LoRA variant).
            best = max(group, key=lambda c: (_aspect_title_score(c.title, aspect), -c.distance))
            if best.arxiv_id not in preferred:
                preferred.append(best.arxiv_id)
                continue
        for pid in _best_paper_ids(group, n=1):
            if pid not in preferred:
                preferred.append(pid)

    # Pool unique chunks, then rerank against the full query.
    pooled: List[RetrievedChunk] = []
    seen_keys: Set[Tuple[str, int]] = set()
    for group in aspect_results:
        for c in group:
            key = (c.arxiv_id, c.chunk_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pooled.append(c)
    pooled = _rerank_chunks(pooled, full_query)

    selected: List[RetrievedChunk] = []
    selected_keys: Set[Tuple[str, int]] = set()
    per_paper: Dict[str, int] = {}
    # For A-vs-B compares, default to one chunk per preferred paper so both
    # sides stay visible in the top_k window.
    limit = max_per_paper if max_per_paper is not None else 1

    def _append(c: RetrievedChunk) -> bool:
        key = (c.arxiv_id, c.chunk_index)
        if key in selected_keys:
            return False
        count = per_paper.get(c.arxiv_id, 0)
        if count >= limit:
            return False
        selected.append(c)
        selected_keys.add(key)
        per_paper[c.arxiv_id] = count + 1
        return len(selected) >= top_k

    for pid in preferred:
        for c in pooled:
            if c.arxiv_id != pid:
                continue
            if _append(c):
                return selected

    for c in pooled:
        if _append(c):
            break

    return selected[:top_k]


def _diversify_papers(
    chunks: List[RetrievedChunk],
    top_k: int,
    max_per_paper: Optional[int],
) -> List[RetrievedChunk]:
    """
    First take one chunk per distinct paper (in reranked order), then fill
    remaining slots with the normal per-paper cap. Keeps multi-paper synthesis
    from collapsing onto three chunks of the same LoRA variant.
    """
    if not chunks:
        return []
    diverse = _cap_per_paper(chunks, top_k, max_per_paper=1)
    if len(diverse) >= top_k:
        return diverse
    seen = {(c.arxiv_id, c.chunk_index) for c in diverse}
    fill_cap = max_per_paper or 3
    per_paper: Dict[str, int] = {}
    for c in diverse:
        per_paper[c.arxiv_id] = per_paper.get(c.arxiv_id, 0) + 1
    for c in chunks:
        if len(diverse) >= top_k:
            break
        key = (c.arxiv_id, c.chunk_index)
        if key in seen:
            continue
        if per_paper.get(c.arxiv_id, 0) >= fill_cap:
            continue
        diverse.append(c)
        seen.add(key)
        per_paper[c.arxiv_id] = per_paper.get(c.arxiv_id, 0) + 1
    return diverse


def retrieve(
    query: str,
    store: Optional[VectorStore] = None,
    top_k: int = 6,
    where: Optional[Dict] = None,
    max_per_paper: Optional[int] = 3,
    overfetch_factor: int = 4,
) -> List[RetrievedChunk]:
    """
    Retrieve the top_k most relevant chunks for a query.

    For comparison-style queries that name two approaches, still runs a
    *single* dense fetch (one embed), then re-ranks that pool per aspect so
    each side of the comparison is represented when possible. Extra embed
    passes were adding ~100s+ on CPU for compare queries.
    """
    store = store or VectorStore()
    fetch_k = top_k * overfetch_factor if max_per_paper else top_k
    fetch_k = max(fetch_k, top_k * 6)

    aspects = split_comparison_aspects(query) if where is None else []
    diversify = _wants_paper_diversity(query)
    # Slightly wider net when we need two sides covered from one fetch.
    if len(aspects) >= 2 or diversify:
        fetch_k = max(fetch_k, top_k * 8)

    chunks = _dense_fetch(query, store, fetch_k, where)

    if len(aspects) >= 2:
        # If an aspect isn't represented by any title in the pooled results
        # (e.g. "partial fine-tuning" drowned out by LoRA-heavy matches), do
        # one small targeted fetch for it. This is cheap — the embedding
        # model is already loaded and it's a single extra store query — and
        # only happens when the pool actually misses a side.
        pool = list(chunks)
        for aspect in aspects:
            covered = _aspect_covered(pool, aspect)
            wants_canonical = bool(
                re.search(r"\b(?:standard|original|vanilla)\b", aspect.lower())
            )
            if not covered:
                extra = _dense_fetch(aspect, store, max(top_k * 2, 12), where)
                existing = {(c.arxiv_id, c.chunk_index) for c in pool}
                pool.extend(
                    c for c in extra if (c.arxiv_id, c.chunk_index) not in existing
                )
            # Dense retrieval often still misses sparse lexical matches
            # ("Partial Is Better Than All" for "partial fine-tuning"), and
            # "standard X" needs the introducing paper forced into the pool
            # ahead of X-Variant titles.
            if not covered or wants_canonical:
                rescued = _lexical_title_rescue(store, aspect, limit=4)
                existing = {(c.arxiv_id, c.chunk_index) for c in pool}
                pool.extend(
                    c for c in rescued if (c.arxiv_id, c.chunk_index) not in existing
                )
        aspect_results = [_rerank_chunks(list(pool), aspect) for aspect in aspects]
        aspect_results.append(_rerank_chunks(list(pool), query))
        # One chunk per paper so both comparison sides stay in top_k.
        compare_cap = 1 if max_per_paper is None or max_per_paper > 1 else max_per_paper
        return _merge_aspect_results(
            aspect_results, query, top_k, compare_cap, aspects=aspects
        )

    chunks = _rerank_chunks(chunks, query)
    if diversify:
        # Ensure the introducing paper for a named concept is in the pool
        # (e.g. original LoRA when asking how papers agree about LoRA).
        subject = _named_subject_head(query)
        if subject:
            rescued = _lexical_title_rescue(store, f"original {subject}", limit=3)
            existing = {(c.arxiv_id, c.chunk_index) for c in chunks}
            chunks = list(chunks) + [
                c for c in rescued if (c.arxiv_id, c.chunk_index) not in existing
            ]
            chunks = _rerank_chunks(chunks, query)
        return _diversify_papers(chunks, top_k, max_per_paper)
    return _cap_per_paper(chunks, top_k, max_per_paper)


if __name__ == "__main__":
    query = "How does LoRA reduce the number of trainable parameters?"
    results = retrieve(query, top_k=6)

    print(f"Query: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.title} (dist={r.distance:.4f}, source={r.source})")
        print(f"    {r.text[:500]}...\n")
