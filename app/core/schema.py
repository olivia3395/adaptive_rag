from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    data_dir: str = Field(..., description="Directory containing .txt or .md files")


class QueryRequest(BaseModel):
    question: str
    mode: str = "adaptive"  # vanilla | graph | adaptive
    top_k: int = 4
    max_iterations: int = 2


class RetrievedItem(BaseModel):
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class ReflectionResult(BaseModel):
    label: str
    score: float
    explanation: str
    unsupported_claims: List[str]


class QueryResponse(BaseModel):
    mode: str
    route: str
    answer: str
    iterations_used: int
    retrieved: List[RetrievedItem]
    reflection: ReflectionResult
    follow_up_queries: List[str]
    debug: Optional[Dict[str, Any]] = None
