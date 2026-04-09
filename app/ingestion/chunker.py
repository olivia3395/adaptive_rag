from __future__ import annotations

from typing import List
from app.utils.text_utils import normalize_whitespace


class TextChunker:
    def __init__(self, chunk_size: int = 650, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        text = normalize_whitespace(text)
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks
