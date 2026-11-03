"""
Standalone arXiv search tool.
Given a topic string, returns top N papers with title/abstract/id/url.
No agent framework dependency — pure function, testable in isolation.
"""

import feedparser
import re
import requests
import time
from dataclasses import dataclass, asdict
from typing import List

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_MAX_RETRIES = 3
ARXIV_RETRY_BASE_WAIT = 2  # seconds

# arXiv's search_query uses (), ", and : for boolean grouping / field syntax.
# A raw natural-language question like "What is RAG (retrieval-augmented
# generation)?" contains unbalanced/stray instances of these and arXiv's
# parser 500s on it — this isn't a rate-limit issue so retries don't help.
# Strip anything that isn't alphanumeric/space/hyphen before building the query.
_ARXIV_UNSAFE_CHARS = re.compile(r"[^\w\s\-\.]")


def sanitize_arxiv_topic(topic: str) -> str:
    cleaned = _ARXIV_UNSAFE_CHARS.sub(" ", topic)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass
class ArxivPaper:
    arxiv_id: str          # e.g. "2106.09685"
    title: str
    abstract: str
    authors: List[str]
    url: str               # abs page, e.g. https://arxiv.org/abs/2106.09685
    pdf_url: str
    published: str         # ISO date string
    updated: str

    def to_dict(self):
        return asdict(self)


def _arxiv_request_with_retry(params: dict, max_retries: int = ARXIV_MAX_RETRIES, base_wait: int = ARXIV_RETRY_BASE_WAIT):
    """
    Make an arXiv API request with exponential backoff retry.

    arXiv often returns 503 (Service Unavailable) when rate-limited.
    This function retries with exponential backoff to handle transient failures.

    Args:
        params: Query parameters for the API request
        max_retries: Number of retries (default 3)
        base_wait: Initial wait time in seconds (default 2)

    Returns:
        Response object

    Raises:
        requests.exceptions.RequestException: After max retries exhausted
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(ARXIV_API_URL, params=params, timeout=60)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries:
                raise
            wait_time = base_wait * (2 ** attempt)  # exponential backoff
            print(f"[arxiv_search] retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries + 1}, error: {e})")
            time.sleep(wait_time)
        except requests.HTTPError as e:
            # 429 (Too Many Requests) or 503 (Service Unavailable) are rate-limit responses
            if response.status_code in (429, 503) and attempt < max_retries:
                wait_time = base_wait * (2 ** attempt)
                print(f"[arxiv_search] HTTP {response.status_code}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(wait_time)
            else:
                raise


def search_arxiv(
    topic: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[ArxivPaper]:
    """
    Query the arXiv API for a topic and return structured paper results.

    Args:
        topic: search string, e.g. "LoRA fine-tuning". Sanitized internally —
            arXiv's query syntax treats (), ", and : as boolean/field
            operators, so raw natural-language questions containing them
            can cause arXiv's parser to 500. Pass a topic phrase, not a
            full question, for best results.
        max_results: number of results to return (arXiv caps at 2000/request)
        sort_by: 'relevance' | 'lastUpdatedDate' | 'submittedDate'
        sort_order: 'ascending' | 'descending'

    Returns:
        List of ArxivPaper objects, empty list on no results.

    Raises:
        ValueError: if topic is empty (including after sanitization).
        requests.exceptions.RequestException on network/HTTP failure.
    """
    if not topic.strip():
        raise ValueError("Topic cannot be empty.")

    clean_topic = sanitize_arxiv_topic(topic)
    if not clean_topic:
        raise ValueError(
            f"Topic {topic!r} contained no usable search terms after sanitization."
        )

    query = f"all:{clean_topic}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    # Use retry wrapper to handle rate-limiting (503) and connection issues
    response = _arxiv_request_with_retry(params)

    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        # entry.id looks like http://arxiv.org/abs/2106.09685v1
        raw_id = entry.id.split("/abs/")[-1]
        arxiv_id = raw_id.rsplit("v", 1)[0] if "v" in raw_id else raw_id

        pdf_link = ""
        for link in entry.links:
            if getattr(link, "title", "") == "pdf" or link.get("type") == "application/pdf":
                pdf_link = link.href
                break
        if not pdf_link:
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=entry.title.replace("\n", " ").strip(),
                abstract=entry.summary.replace("\n", " ").strip(),
                authors=[a.name for a in entry.authors] if hasattr(entry, "authors") else [],
                url=entry.id,
                pdf_url=pdf_link,
                published=entry.published if hasattr(entry, "published") else "",
                updated=entry.updated if hasattr(entry, "updated") else "",
            )
        )

    return papers


if __name__ == "__main__":
    # Quick manual smoke test — run: python tools/arxiv_search.py
    results = search_arxiv("LoRA fine-tuning", max_results=5)
    for i, p in enumerate(results, 1):
        print(f"\n[{i}] {p.title}")
        print(f"    id: {p.arxiv_id}  url: {p.url}")
        print(f"    authors: {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"    abstract: {p.abstract[:150]}...")