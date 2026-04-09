from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ChunkRecord:
    chunk_id: str
    content: str
    metadata: Dict


class DocumentStore:
    def __init__(self) -> None:
        self.records: List[ChunkRecord] = []
        self.by_id: Dict[str, ChunkRecord] = {}

    def add(self, record: ChunkRecord) -> None:
        self.records.append(record)
        self.by_id[record.chunk_id] = record

    def get(self, chunk_id: str) -> ChunkRecord:
        return self.by_id[chunk_id]

    def all(self) -> List[ChunkRecord]:
        return list(self.records)
