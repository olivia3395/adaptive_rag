from __future__ import annotations

from typing import List
from app.utils.text_utils import content_tokens


class FollowUpQueryGenerator:
    def generate(self, question: str, draft_answer: str, retrieved_contexts: List[str]) -> str:
        q_tokens = content_tokens(question)
        context_tokens = []
        for c in retrieved_contexts:
            context_tokens.extend(content_tokens(c))
        missing = [t for t in q_tokens if t not in set(context_tokens)]
        hint = " ".join(missing[:5])
        if hint:
            return f"{question} {hint}".strip()
        return f"more evidence for: {question}".strip()
