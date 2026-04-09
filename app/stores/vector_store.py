from __future__ import annotations

from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.matrix = None

    def build(self, ids: List[str], documents: List[str], metadatas: List[Dict], matrix) -> None:
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.matrix = matrix

    def query(self, query_vector, top_k: int = 4) -> List[Dict]:
        if self.matrix is None or len(self.ids) == 0:
            return []
        sims = cosine_similarity(query_vector, self.matrix)[0]
        idxs = np.argsort(-sims)[:top_k]
        out = []
        for idx in idxs:
            out.append(
                {
                    "chunk_id": self.ids[idx],
                    "score": float(sims[idx]),
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                }
            )
        return out
