from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.core.schema import QueryRequest

router = APIRouter()
state = {}


def set_query_pipeline(pipeline) -> None:
    state["pipeline"] = pipeline


@router.post("/query")
def query(req: QueryRequest):
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Query pipeline not initialized")
    return pipeline.run(
        question=req.question,
        mode=req.mode,
        top_k=req.top_k,
        max_iterations=req.max_iterations,
    )
