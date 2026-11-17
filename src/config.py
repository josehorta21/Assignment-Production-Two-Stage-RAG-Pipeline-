"""Load validated, project-root-relative settings without requiring API keys."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    local_llm_model: str = "google/flan-t5-small"
    rewrite_mode: str = "local"
    generation_mode: str = "local"
    # 0.35 was selected on the 20-case educational calibration set (Nov. 16, 2026).
    # It is not validated on an independent held-out sample.
    threshold: float = 0.35
    k1: int = 20
    k2: int = 5
    web_results: int = 3
    docs_dir: Path = PROJECT_ROOT / "docs"
    vector_store_dir: Path = PROJECT_ROOT / ".rag_index"


def _project_path(value: str) -> Path:
    """Resolve relative documentation/cache paths against the project root."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_settings() -> Settings:
    """Load .env from this repository; exported environment variables take priority."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    settings = Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        reranker_model=os.getenv("RERANKER_MODEL", Settings.reranker_model),
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", Settings.local_llm_model),
        rewrite_mode=os.getenv("REWRITE_MODE", Settings.rewrite_mode),
        generation_mode=os.getenv("GENERATION_MODE", Settings.generation_mode),
        threshold=float(os.getenv("CONFIDENCE_THRESHOLD", str(Settings.threshold))),
        k1=int(os.getenv("K1", str(Settings.k1))),
        k2=int(os.getenv("K2", str(Settings.k2))),
        web_results=int(os.getenv("WEB_RESULTS", str(Settings.web_results))),
        docs_dir=_project_path(os.getenv("DOCS_DIR", "docs")),
        vector_store_dir=_project_path(os.getenv("VECTOR_STORE_DIR", ".rag_index")),
    )
    if not 0.0 <= settings.threshold <= 1.0:
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
    if settings.k1 < 1 or settings.k2 < 1 or settings.k2 > settings.k1:
        raise ValueError("Require 1 <= K2 <= K1")
    return settings
