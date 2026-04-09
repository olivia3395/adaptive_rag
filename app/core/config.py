from __future__ import annotations

import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    app_name: str = "AdaptiveRAG"
    host: str = "0.0.0.0"
    port: int = 8000

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    chunk_size: int = 650
    chunk_overlap: int = 120
    top_k: int = 4
    max_iterations: int = 2
    grounding_threshold: float = 0.55


settings = Settings()
