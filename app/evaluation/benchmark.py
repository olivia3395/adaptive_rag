from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import re


@dataclass
class Example:
    question: str
    gold_answer: str


class BenchmarkRunner:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\W+", " ", text.lower()).strip()

    @staticmethod
    def _token_f1(pred: str, gold: str) -> float:
        p = set(BenchmarkRunner._normalize(pred).split())
        g = set(BenchmarkRunner._normalize(gold).split())
        if not p or not g:
            return 0.0
        inter = len(p & g)
        if inter == 0:
            return 0.0
        precision = inter / len(p)
        recall = inter / len(g)
        return 2 * precision * recall / (precision + recall)

    def evaluate(self, examples: List[Example], mode: str) -> Dict:
        hits = 0
        results = []
        for ex in examples:
            out = self.pipeline.run(ex.question, mode=mode)
            f1 = self._token_f1(out["answer"], ex.gold_answer)
            correct = f1 >= 0.35
            hits += int(correct)
            results.append({
                "question": ex.question,
                "gold": ex.gold_answer,
                "pred": out["answer"],
                "f1": round(f1, 4),
                "correct": correct,
                "reflection": out["reflection"],
            })
        return {
            "mode": mode,
            "accuracy": hits / max(1, len(examples)),
            "num_examples": len(examples),
            "results": results,
        }
