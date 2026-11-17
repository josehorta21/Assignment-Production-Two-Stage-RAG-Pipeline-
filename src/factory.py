"""Build a fully local RAG instance, or inject test doubles for automated QA."""
from __future__ import annotations

from .config import Settings, load_settings
from .generator import GroundedGenerator
from .pipeline import RAGPipeline
from .reranker import CrossEncoderReranker
from .retriever import DenseRetriever
from .rewriter import QueryRewriter
from .web_fallback import WebFallback
from .local_llm import LocalTextModel


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    settings = settings or load_settings()
    llm = (LocalTextModel(settings.local_llm_model)
           if (settings.rewrite_mode == "local" or settings.generation_mode == "local") else None)
    return RAGPipeline(
        rewriter=QueryRewriter(llm, mode=settings.rewrite_mode),
        retriever=DenseRetriever(settings.docs_dir, settings.vector_store_dir, settings.embedding_model),
        reranker=CrossEncoderReranker(settings.reranker_model),
        web_fallback=WebFallback(max_results=settings.web_results),
        generator=GroundedGenerator(llm=llm, mode=settings.generation_mode),
        threshold=settings.threshold, k1=settings.k1, k2=settings.k2,
    )
