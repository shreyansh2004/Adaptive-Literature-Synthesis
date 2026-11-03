"""
Embedding backend.

Switch between OpenAI embeddings and a local embedding model.
"""

import os
from typing import List

USE_OPENAI = False

if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI()   # reads OPENAI_API_KEY from env

    def embed_texts(texts: List[str]) -> List[List[float]]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

else:
    # Option B: local sentence-transformers
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed_texts(texts: List[str]) -> List[List[float]]:
        return _model.encode(texts, convert_to_numpy=True).tolist()