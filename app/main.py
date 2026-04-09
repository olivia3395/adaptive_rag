from __future__ import annotations

from fastapi import FastAPI

from app.core.config import settings
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router, set_ingestion_pipeline
from app.api.routes_query import router as query_router, set_query_pipeline
from app.embeddings.tfidf_embedder import TfidfEmbedder
from app.stores.vector_store import InMemoryVectorStore
from app.stores.document_store import DocumentStore
from app.retrieval.graph_rag import GraphIndex
from app.retrieval.dense_retriever import DenseRetriever
from app.llm.generator import LLMGenerator
from app.pipelines.ingestion_pipeline import IngestionPipeline
from app.pipelines.adaptive_rag_pipeline import AdaptiveRAGPipeline


app = FastAPI(title=settings.app_name)

document_store = DocumentStore()
embedder = TfidfEmbedder()
vector_store = InMemoryVectorStore()
graph_index = GraphIndex()
retriever = DenseRetriever(embedder, vector_store)
retriever.document_store = document_store  # lightweight dependency injection
generator = LLMGenerator(settings.openai_api_key, settings.openai_model)

ingestion_pipeline = IngestionPipeline(document_store, embedder, vector_store, graph_index)
query_pipeline = AdaptiveRAGPipeline(retriever, graph_index, generator)

set_ingestion_pipeline(ingestion_pipeline)
set_query_pipeline(query_pipeline)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
