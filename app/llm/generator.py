from __future__ import annotations

from typing import List, Optional
import os

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from app.utils.text_utils import content_tokens, overlap_ratio, simple_sentences


class LLMGenerator:
    def __init__(self, api_key: str = "", model: str = "gpt-4.1-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if (self.api_key and OpenAI is not None) else None

    def _fallback_answer(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "I do not have enough evidence to answer confidently."
        q_toks = content_tokens(question)
        scored = []
        for ctx in contexts:
            best_sent = ""
            best_score = -1.0
            for sent in simple_sentences(ctx):
                score = overlap_ratio(q_toks, content_tokens(sent))
                if score > best_score:
                    best_score = score
                    best_sent = sent
            scored.append((best_score, best_sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = []
        seen = set()
        for _, s in scored:
            key = s.strip().lower()
            if s and key not in seen:
                top.append(s)
                seen.add(key)
            if len(top) == 3:
                break
        if not top:
            return "I found documents, but none clearly support a confident answer."
        return " ".join(top)

    def answer(self, question: str, contexts: List[str]) -> str:
        if self.client is None:
            return self._fallback_answer(question, contexts)

        evidence = "\n\n".join(f"[DOC {i+1}] {c}" for i, c in enumerate(contexts[:8]))
        prompt = (
            "You are a grounded QA assistant. Answer only from the evidence. "
            "If evidence is insufficient, say so explicitly.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        resp = self.client.responses.create(model=self.model, input=prompt)
        return resp.output_text
