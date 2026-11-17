"""Environment-driven settings without mandatory API keys."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    local_llm_model: str = "google/flan-t5-small"
    rewrite_mode: str = "local"
    generation_mode: str = "local"
    threshold: float = 0.55
    k1: int = 20
    k2: int = 5
    web_results: int = 3
    docs_dir: Path = Path("docs")
    vector_store_dir: Path = Path(".rag_index")


def load_settings() -> Settings:
    from dotenv import load_dotenv
    load_dotenv()
    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        reranker_model=os.getenv("RERANKER_MODEL", Settings.reranker_model),
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", Settings.local_llm_model),
        rewrite_mode=os.getenv("REWRITE_MODE", Settings.rewrite_mode),
        generation_mode=os.getenv("GENERATION_MODE", Settings.generation_mode),
        threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.55")),
        k1=int(os.getenv("K1", "20")),
        k2=int(os.getenv("K2", "5")),
        web_results=int(os.getenv("WEB_RESULTS", "3")),
        docs_dir=Path(os.getenv("DOCS_DIR", "docs")),
        vector_store_dir=Path(os.getenv("VECTOR_STORE_DIR", ".rag_index")),
    )
