"""
Utilities for splitting research papers into token-aware chunks.
"""

from typing import List
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    """
    Token-aware sliding-window chunking.
    Splits on paragraphs first; long paragraphs get token-split with overlap.

    Strategy:
    - Split by paragraphs first.
    - If a paragraph exceeds chunk_size, split by tokens.
    """

    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buffer = ""
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = len(ENCODING.encode(para))

        if para_tokens > chunk_size:
            # Flush current buffer first
            if buffer:
                chunks.append(buffer)
                buffer = ""
                buffer_tokens = 0
            # Token-split the oversized paragraph directly
            tokens = ENCODING.encode(para)
            start = 0
            while start < len(tokens):
                end = start + chunk_size
                chunk = ENCODING.decode(tokens[start:end])
                chunks.append(chunk)
                start = end - overlap
            continue

        if buffer_tokens + para_tokens <= chunk_size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
            buffer_tokens += para_tokens
        else:
            chunks.append(buffer)
            buffer = para
            buffer_tokens = para_tokens

    if buffer:
        chunks.append(buffer)

    return chunks