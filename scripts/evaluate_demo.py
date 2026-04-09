from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings.tfidf_embedder import TfidfEmbedder
from app.stores.vector_store import InMemoryVectorStore
from app.stores.document_store import DocumentStore
from app.retrieval.graph_rag import GraphIndex
from app.retrieval.dense_retriever import DenseRetriever
from app.llm.generator import LLMGenerator
from app.pipelines.ingestion_pipeline import IngestionPipeline
from app.pipelines.adaptive_rag_pipeline import AdaptiveRAGPipeline
from app.evaluation.benchmark import BenchmarkRunner, Example


def build_pipeline():
    document_store = DocumentStore()
    embedder = TfidfEmbedder()
    vector_store = InMemoryVectorStore()
    graph_index = GraphIndex()
    retriever = DenseRetriever(embedder, vector_store)
    retriever.document_store = document_store
    generator = LLMGenerator()
    ingestion = IngestionPipeline(document_store, embedder, vector_store, graph_index)
    ingestion.ingest("./data/demo")
    return AdaptiveRAGPipeline(retriever, graph_index, generator)


def main():
    pipeline = build_pipeline()
    runner = BenchmarkRunner(pipeline)
    examples = [
        Example(
            question="How does adaptive RAG differ from vanilla RAG?",
            gold_answer="Adaptive RAG can decide whether to retrieve, retrieve iteratively, and reflect on grounding.",
        ),
        Example(
            question="What is the role of self-reflection in adaptive RAG?",
            gold_answer="It checks whether the answer is supported by documents and can trigger retry or refusal.",
        ),
    ]
    for mode in ["vanilla", "graph", "adaptive"]:
        result = runner.evaluate(examples, mode=mode)
        print(f"{mode}: accuracy={result['accuracy']:.2f} on {result['num_examples']} examples")


if __name__ == "__main__":
    main()
