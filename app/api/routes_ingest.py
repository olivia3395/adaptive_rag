from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.core.schema import IngestRequest

router = APIRouter()
state = {}


def set_ingestion_pipeline(pipeline) -> None:
    state["pipeline"] = pipeline


@router.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Ingestion pipeline not initialized")
    n = pipeline.ingest(req.data_dir)
    return {"message": "ingestion complete", "num_chunks": n, "data_dir": req.data_dir}
