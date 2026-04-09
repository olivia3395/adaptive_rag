from __future__ import annotations

import re
from typing import Iterable, List, Set

STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "as", "at",
    "by", "from", "into", "about", "what", "which", "who", "whom", "when", "where", "why",
    "how", "do", "does", "did", "can", "could", "should", "would", "may", "might", "will",
    "shall", "i", "you", "he", "she", "they", "we", "them", "his", "her", "their", "our",
    "your", "my", "me", "us", "than", "then", "there", "here", "also", "not", "no", "yes",
    "more", "most", "less", "least", "very", "much", "many", "some", "any"
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def simple_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_whitespace(text))
    return [p.strip() for p in parts if p.strip()]


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_-]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def overlap_ratio(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / max(1, len(sa))
