"""
Vector store wrapper around ChromaDB.
Keeps ingest.py and retrieve.py decoupled from Chroma's API specifics.
"""

import chromadb
# from chromadb.config import Settings
from typing import List, Dict, Optional

CHROMA_PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "papers"


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ):
        """Add or upsert chunks into the collection."""
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ):
        """Return top_k most similar chunks, optionally filtered by metadata."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

    def count(self) -> int:
        return self.collection.count()

    def paper_already_ingested(self, arxiv_id: str) -> bool:
        """Check if any chunks from this paper already exist (idempotency check)."""
        result = self.collection.get(where={"arxiv_id": arxiv_id}, limit=1)
        return len(result["ids"]) > 0

    def delete_paper(self, arxiv_id: str):
        """Remove all chunks belonging to a given paper (used before re-ingest)."""
        self.collection.delete(where={"arxiv_id": arxiv_id})

    def close(self):
        """Release Chroma resources (important on Windows)."""
        self.collection = None
        self.client = None

    def __del__(self):
        self.close()