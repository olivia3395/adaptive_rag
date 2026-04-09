from __future__ import annotations

from typing import Dict, List

from app.core.config import settings
from app.retrieval.query_classifier import QueryClassifier
from app.retrieval.followup_generator import FollowUpQueryGenerator
from app.llm.reflection import ReflectionChecker


class AdaptiveRAGPipeline:
    def __init__(self, retriever, graph_index, generator) -> None:
        self.retriever = retriever
        self.graph_index = graph_index
        self.generator = generator
        self.classifier = QueryClassifier()
        self.followup = FollowUpQueryGenerator()
        self.reflection = ReflectionChecker()

    def _direct_answer(self, question: str) -> str:
        return self.generator.answer(question, [])

    def _graph_expand(self, question: str, hits: List[Dict]) -> List[Dict]:
        seed_ids = [h["chunk_id"] for h in hits]
        expanded_ids = self.graph_index.expand(question, seed_ids, hops=1)
        existing = set(seed_ids)
        merged = list(hits)
        for cid in expanded_ids:
            if cid in existing:
                continue
            try:
                rec = self.retriever.document_store.get(cid)
            except Exception:
                continue
            merged.append(
                {
                    "chunk_id": cid,
                    "score": 0.4,
                    "content": rec.content,
                    "metadata": rec.metadata,
                }
            )
        return merged

    def run(self, question: str, mode: str = "adaptive", top_k: int = 4, max_iterations: int = 2) -> Dict:
        max_iterations = max(1, min(max_iterations, 4))
        route = self.classifier.classify(question)
        followups: List[str] = []
        iterations_used = 0

        if mode == "vanilla":
            route_name = "rag"
        elif mode == "graph":
            route_name = "rag"
        elif mode == "adaptive":
            route_name = route.route
        else:
            route_name = "rag"

        if route_name == "direct":
            answer = self._direct_answer(question)
            reflection = self.reflection.check(answer, [])
            return {
                "mode": mode,
                "route": route_name,
                "answer": answer,
                "iterations_used": 0,
                "retrieved": [],
                "reflection": reflection,
                "follow_up_queries": followups,
                "debug": {"route_reason": route.reason},
            }

        current_query = question
        all_hits: List[Dict] = []
        answer = "I do not have enough evidence to answer confidently."
        reflection = {"label": "hallucinated", "score": 0.0, "explanation": "No evidence.", "unsupported_claims": []}

        for step in range(max_iterations):
            iterations_used += 1
            hits = self.retriever.retrieve(current_query, top_k=top_k)
            if mode in {"graph", "adaptive"}:
                hits = self._graph_expand(question, hits)

            dedup = {}
            for h in all_hits + hits:
                dedup[h["chunk_id"]] = h
            all_hits = sorted(dedup.values(), key=lambda x: x["score"], reverse=True)
            contexts = [h["content"] for h in all_hits[:8]]
            answer = self.generator.answer(question, contexts)
            reflection = self.reflection.check(answer, contexts)

            if mode == "vanilla":
                break
            if reflection["label"] == "grounded":
                break
            if step < max_iterations - 1:
                current_query = self.followup.generate(question, answer, contexts)
                followups.append(current_query)

        if reflection["label"] == "hallucinated":
            answer = "I cannot answer confidently from the retrieved evidence. Please provide more context or documents."

        return {
            "mode": mode,
            "route": route_name,
            "answer": answer,
            "iterations_used": iterations_used,
            "retrieved": all_hits[:10],
            "reflection": reflection,
            "follow_up_queries": followups,
            "debug": {"route_reason": route.reason},
        }
