from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set


class GraphIndex:
    def __init__(self) -> None:
        self.entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)

    def add_chunk(self, chunk_id: str, content: str) -> None:
        entities = self.extract_entities(content)
        self.chunk_to_entities[chunk_id].update(entities)
        for ent in entities:
            self.entity_to_chunks[ent].add(chunk_id)

    @staticmethod
    def extract_entities(text: str) -> Set[str]:
        raw = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)
        return {x.strip() for x in raw}

    def expand(self, question: str, seed_chunk_ids: List[str], hops: int = 1) -> List[str]:
        entities = set(self.extract_entities(question))
        for cid in seed_chunk_ids:
            entities |= self.chunk_to_entities.get(cid, set())

        expanded_chunks: Set[str] = set(seed_chunk_ids)
        frontier = set(entities)
        for _ in range(hops):
            next_entities: Set[str] = set()
            for ent in frontier:
                related_chunks = self.entity_to_chunks.get(ent, set())
                expanded_chunks |= related_chunks
                for cid in related_chunks:
                    next_entities |= self.chunk_to_entities.get(cid, set())
            frontier = next_entities
        return list(expanded_chunks)
