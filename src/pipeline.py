"""End-to-end resilient retrieval pipeline with explicit branch diagnostics."""
from __future__ import annotations

import re
import time
from .schemas import RAGResponse, RetrievedChunk
from .reranker import route_confidence

_FRESHNESS = re.compile(r"\b(today|yesterday|this week|right now|currently|latest|breaking|announced)\b", re.I)


class RAGPipeline:
    def __init__(self, rewriter, retriever, reranker, web_fallback, generator,
                 *, threshold: float = 0.35, k1: int = 20, k2: int = 5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between zero and one")
        if k1 < 1 or k2 < 1 or k2 > k1:
            raise ValueError("Require 1 <= K2 <= K1")
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.web_fallback = web_fallback
        self.generator = generator
        self.threshold, self.k1, self.k2 = threshold, k1, k2

    def run(self, query: str, history: list[str] | None = None,
            *, force_web: bool = False) -> RAGResponse:
        begin = time.perf_counter()
        transformed = self.rewriter.rewrite(query, history or [])
        stage1: list[RetrievedChunk] = self.retriever.search(transformed, k=self.k1)
        stage2: list[RetrievedChunk] = self.reranker.rerank(transformed, stage1, k2=self.k2)
        confidence, accepted = route_confidence(stage2, self.threshold)
        recent = bool(_FRESHNESS.search(query))
        warnings: list[str] = []
        if force_web:
            accepted, reason = False, "Explicit web-search override requested by user."
        elif recent:
            accepted, reason = False, "Freshness-sensitive question; internal static docs may be stale."
        elif not stage1:
            accepted, reason = False, "No internal candidates were retrieved."
        elif not accepted:
            reason = f"Top sigmoid-transformed cross-encoder score {confidence:.3f} < {self.threshold:.2f}."
        else:
            reason = f"Top sigmoid-transformed cross-encoder score {confidence:.3f} >= {self.threshold:.2f}."
        if accepted:
            route, selected = "INTERNAL_RAG", stage2
        else:
            route, selected = "WEB_FALLBACK", self.web_fallback.search(transformed)
            if not selected:
                warnings.append(getattr(self.web_fallback, "last_error", None) or "Live search returned no usable results.")
        answer, citations = self.generator.synthesize(transformed, selected)
        if route == "WEB_FALLBACK" and selected:
            warnings.append("Web results contain previews only; publication dates and underlying claims are not verified.")
        return RAGResponse(
            answer=answer, citations=citations, route_taken=route,
            confidence_score=confidence, transformed_query=transformed, original_query=query,
            gate_threshold=self.threshold, gate_reason=reason,
            stage1=stage1, stage2=stage2, sources=selected[:3],
            warnings=[v for v in warnings if v],
            latency_ms=round((time.perf_counter() - begin) * 1000, 2),
            generation_mode=getattr(self.generator, "last_mode", "extractive"),
        )
