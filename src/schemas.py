"""Typed models for retrieval, routing, citations, and telemetry."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    score: float = 0.0
    source: Literal["internal_doc", "web"] = "internal_doc"
    title: str = ""
    url: str | None = None
    stage1_rank: int | None = None
    stage2_rank: int | None = None
    raw_logit: float | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    route_taken: Literal["INTERNAL_RAG", "WEB_FALLBACK"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    transformed_query: str
    original_query: str
    gate_threshold: float
    gate_reason: str
    stage1: list[RetrievedChunk] = Field(default_factory=list)
    stage2: list[RetrievedChunk] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    generation_mode: str = "extractive"
