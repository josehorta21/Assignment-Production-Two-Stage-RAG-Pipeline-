"""Deterministic test doubles: no live model downloads or network requests."""
from __future__ import annotations
import pytest
from src.schemas import RetrievedChunk
from src.rewriter import QueryRewriter
from src.generator import GroundedGenerator
from src.pipeline import RAGPipeline
from src.web_fallback import WebFallback


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    def search(self, query, k=20):
        self.calls += 1
        return self.candidates[:k]


class FakeReranker:
    def __init__(self, confidence):
        self.confidence = confidence

    def rerank(self, query, candidates, k2=5):
        selected = []
        for i, candidate in enumerate(candidates[:k2], 1):
            sample = candidate.model_copy(deep=True)
            sample.stage2_rank = i
            sample.score = self.confidence
            sample.raw_logit = 1.5
            selected.append(sample)
        return selected


class FakeWebSearch:
    last_error = None

    def __init__(self, results=None):
        self.results = results or []
        self.calls = 0

    def search(self, query):
        self.calls += 1
        return self.results


@pytest.fixture
def internal_chunk():
    return RetrievedChunk(
        chunk_id="01_rolling_updates.md::configuration-reference",
        content="maxUnavailable defaults to 25% in Rolling Update Deployment.",
        title="Rolling Update Deployment", source="internal_doc",
        score=0.86, stage1_rank=1, metadata={"file": "01_rolling_updates.md"},
    )


@pytest.fixture
def web_chunk():
    return RetrievedChunk(
        chunk_id="web:1", content="A service disruption was reported by the provider.",
        source="web", url="https://example.org/service-status",
        title="Status page",
    )


@pytest.fixture
def pipeline_factory(internal_chunk, web_chunk):
    def create(confidence=0.81, web_results=None, reranker=None, retriever=None, generator=None):
        return RAGPipeline(
            rewriter=QueryRewriter(),
            retriever=retriever or FakeRetriever([internal_chunk]),
            reranker=reranker or FakeReranker(confidence),
            web_fallback=FakeWebSearch([web_chunk] if web_results is None else web_results),
            generator=generator or GroundedGenerator(), threshold=0.55, k1=20, k2=5,
        )
    return create
