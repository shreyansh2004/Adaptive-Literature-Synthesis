"""
Ingestion pipeline: 

Loads fetched papers from data/raw/,
chunks them,
embeds them,
and stores them inside ChromaDB.
"""

import os
import re
import glob
import json
from typing import List, Dict

from rag.chunking import chunk_text
from rag.embeddings import embed_texts
from rag.store import VectorStore


RAW_DATA_DIR = "data/raw"

def build_chunk_metadata(paper: Dict, chunk_index: int, source: str) -> Dict:
    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "authors": ", ".join(paper["authors"]),
        "published": paper["published"],
        "updated": paper["updated"],
        "url": paper["url"],
        "chunk_index": chunk_index,
        "source": source,
    }

def clean_paper_text(text: str) -> str:
    """Strip common low-value sections before chunking."""
    # Cut everything from the References section onward
    text = re.split(r'\n\s*References\s*\n', text, maxsplit=1, flags=re.IGNORECASE)[0]
    # Also drop a likely title/author/affiliation header block:
    # heuristic — if the text starts with the paper title repeated (common in PDF extraction)
    # followed by short author-name lines, skip past the first "Abstract" marker if present.
    abstract_match = re.search(r'\n\s*Abstract\s*\n', text, flags=re.IGNORECASE)
    if abstract_match and abstract_match.start() < 1500:  # only if header is short/near the top
        text = text[abstract_match.start():]

    return text

def ingest_paper(paper: Dict, store: VectorStore, force: bool = False) -> int:
    """
    Chunk + embed + store a single paper (loaded from its raw JSON dict).
    Returns number of chunks added. Skips if already ingested unless force=True.
    """
    arxiv_id = paper["arxiv_id"]

    if not force and store.paper_already_ingested(arxiv_id):
        print(f"[skip] {arxiv_id} already ingested.", flush=True)
        return 0
    
    if force:
        store.delete_paper(arxiv_id)  # clear stale chunks before re-adding


    # Prefer full text if available and non-trivial, else fall back to abstract
    source = (
        paper["full_text"] if paper.get("full_text_fetched") else paper["abstract"]
    )
    source_name = "full_text" if paper.get("full_text_fetched") else "abstract"

    source = clean_paper_text(source)
    chunks = chunk_text(source)
    if not chunks:
        print(f"[warn] {arxiv_id} produced 0 chunks, skipping.", flush=True)
        return 0

    embeddings = embed_texts(chunks)
    ids = [f"{arxiv_id}_{i}" for i in range(len(chunks))]
    metadatas = [build_chunk_metadata(paper, i, source_name) for i in range(len(chunks))]

    store.add_chunks(ids=ids, texts=chunks, embeddings=embeddings, metadatas=metadatas)
    print(f"[ingested] {arxiv_id}: {len(chunks)} chunks (source={source_name})", flush=True)
    return len(chunks)


def ingest_all_raw_papers(raw_dir: str = RAW_DATA_DIR, force: bool = False) -> int:
    """Ingest every paper JSON found in data/raw/. Returns total chunks added."""
    store = VectorStore()
    total = 0
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            paper = json.load(f)
        total += ingest_paper(paper, store, force=force)
    print(f"\nDone. Total chunks added this run: {total}. Collection size: {store.count()}")
    return total


if __name__ == "__main__":
    ingest_all_raw_papers()