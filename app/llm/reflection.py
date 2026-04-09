from __future__ import annotations

from typing import Dict, List
from app.utils.text_utils import content_tokens, overlap_ratio, simple_sentences


class ReflectionChecker:
    def check(self, answer: str, contexts: List[str]) -> Dict:
        answer_sents = simple_sentences(answer)
        context_sents = []
        for ctx in contexts:
            context_sents.extend(simple_sentences(ctx))

        unsupported: List[str] = []
        sent_scores: List[float] = []
        for sent in answer_sents:
            stoks = content_tokens(sent)
            if not stoks:
                continue
            best = 0.0
            for ctx_sent in context_sents:
                score = overlap_ratio(stoks, content_tokens(ctx_sent))
                if score > best:
                    best = score
            sent_scores.append(best)
            if best < 0.35:
                unsupported.append(sent)

        score = sum(sent_scores) / max(1, len(sent_scores))
        if score >= 0.7 and not unsupported:
            label = "grounded"
            explanation = "Most answer sentences are well-supported by retrieved documents."
        elif score >= 0.45:
            label = "partially_grounded"
            explanation = "Some parts are supported, but a few claims are weakly grounded."
        else:
            label = "hallucinated"
            explanation = "The answer contains claims that are not sufficiently supported by the retrieved documents."

        return {
            "label": label,
            "score": float(round(score, 4)),
            "explanation": explanation,
            "unsupported_claims": unsupported,
        }
