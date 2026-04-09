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
    questions = [
        "How does adaptive RAG differ from vanilla RAG?",
        "Why might a system perform iterative retrieval?",
        "What does self-reflection check for?",
    ]
    for mode in ["vanilla", "graph", "adaptive"]:
        print(f"\n=== MODE: {mode} ===")
        for q in questions:
            out = pipeline.run(q, mode=mode)
            print(f"\nQ: {q}")
            print(f"A: {out['answer']}")
            print(f"Reflection: {out['reflection']}")
            print(f"Iterations: {out['iterations_used']}")


if __name__ == "__main__":
    main()
