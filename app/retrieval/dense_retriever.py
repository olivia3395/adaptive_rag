from __future__ import annotations

from typing import Dict, List


class DenseRetriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        qv = self.embedder.transform([query])
        return self.vector_store.query(qv, top_k=top_k)
