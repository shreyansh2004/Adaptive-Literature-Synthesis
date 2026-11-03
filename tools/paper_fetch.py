"""
Fetch full paper data (metadata + optionally full text) for a given arXiv ID.
Stores raw fetched papers to disk as JSON for later ingestion.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional
from io import BytesIO

import feedparser
import requests
from pypdf import PdfReader

from tools.arxiv_search import ArxivPaper

ARXIV_API_URL = "http://export.arxiv.org/api/query"
RAW_DATA_DIR = "data/raw"


@dataclass
class FetchedPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list
    url: str
    pdf_url: str
    published: str
    updated: str
    full_text: Optional[str] = None
    full_text_fetched: bool = False
    fetch_error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def fetch_paper_metadata(arxiv_id: str) -> ArxivPaper:
    """Fetch title/abstract/authors/etc for a single arXiv ID."""
    params = {"id_list": arxiv_id}
    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()

    feed = feedparser.parse(response.text)
    if not feed.entries:
        raise ValueError(f"No entry found for arxiv_id={arxiv_id}")

    entry = feed.entries[0]
    raw_id = entry.id.split("/abs/")[-1]
    clean_id = raw_id.rsplit("v", 1)[0] if "v" in raw_id else raw_id

    pdf_link = ""
    for link in entry.links:
        if getattr(link, "title", "") == "pdf" or link.get("type") == "application/pdf":
            pdf_link = link.href
            break

    return ArxivPaper(
        arxiv_id=clean_id,
        title=entry.title.replace("\n", " ").strip(),
        abstract=entry.summary.replace("\n", " ").strip(),
        authors=[a.name for a in entry.authors] if hasattr(entry, "authors") else [],
        url=entry.id,
        pdf_url=pdf_link,
        published=entry.published if hasattr(entry, "published") else "",
        updated=entry.updated if hasattr(entry, "updated") else "",
    )


def fetch_pdf_full_text(pdf_url: str, timeout: int = 30) -> str:
    """
    Best-effort full text extraction from the paper's PDF.
    Returns extracted text, or raises on failure (caller should catch).
    """
    response = requests.get(pdf_url, timeout=timeout)
    response.raise_for_status()

    reader = PdfReader(BytesIO(response.content))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")

    return "\n".join(text_parts).strip()


def fetch_paper(arxiv_id: str, include_full_text: bool = True) -> FetchedPaper:
    """
    Fetch a single paper's metadata, optionally with full PDF text.
    Full-text failures are non-fatal — metadata is still returned,
    with fetch_error recording what went wrong.
    """
    meta = fetch_paper_metadata(arxiv_id)

    paper = FetchedPaper(
        arxiv_id=meta.arxiv_id,
        title=meta.title,
        abstract=meta.abstract,
        authors=meta.authors,
        url=meta.url,
        pdf_url=meta.pdf_url,
        published=meta.published,
        updated=meta.updated,
    )

    if include_full_text and paper.pdf_url:
        try:
            paper.full_text = fetch_pdf_full_text(paper.pdf_url)
            paper.full_text_fetched = True
        except Exception as e:
            paper.fetch_error = f"PDF fetch/parse failed: {e}"
            paper.full_text_fetched = False

    return paper


def save_paper(paper: FetchedPaper, out_dir: str = RAW_DATA_DIR) -> str:
    """Save a fetched paper as JSON to data/raw/{arxiv_id}.json. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    safe_id = paper.arxiv_id.replace("/", "_")  # old-style IDs can contain slashes
    path = os.path.join(out_dir, f"{safe_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def load_paper(arxiv_id: str, raw_dir: str = RAW_DATA_DIR) -> Optional[FetchedPaper]:
    """Load a previously fetched paper from disk, if it exists. Returns None if not cached."""
    safe_id = arxiv_id.replace("/", "_")
    path = os.path.join(raw_dir, f"{safe_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FetchedPaper(**data)


def fetch_and_save(arxiv_id: str, include_full_text: bool = True, force: bool = False) -> str:
    """
    Convenience wrapper: skip fetching if already cached on disk (unless force=True).
    Returns the file path.
    """
    if not force:
        cached = load_paper(arxiv_id)
        if cached is not None:
            print(f"[cache hit] {arxiv_id} already fetched, skipping.", flush=True)
            return os.path.join(RAW_DATA_DIR, f"{arxiv_id.replace('/', '_')}.json")

    paper = fetch_paper(arxiv_id, include_full_text=include_full_text)
    path = save_paper(paper)
    print(f"[saved] {arxiv_id} -> {path} (full_text: {paper.full_text_fetched})", flush=True)
    return path


if __name__ == "__main__":
    # Manual smoke test — run: python tools/paper_fetch.py
    test_ids = ["2106.09685", "2005.11401"]  # LoRA, RAG papers
    for aid in test_ids:
        fetch_and_save(aid, include_full_text=True)
        time.sleep(3)  # be polite to arXiv's API