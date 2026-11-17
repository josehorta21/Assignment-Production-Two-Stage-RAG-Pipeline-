"""Verify real re-rank implementation logic without downloading model weights."""
import pytest
from src.reranker import CrossEncoderReranker
from src.schemas import RetrievedChunk


def test_pairwise_reranker_uses_logits_and_reorders():
    class PredictStub:
        def predict(self, pairs, **kwargs):
            assert len(pairs) == 2
            return [-2.0, 4.0]
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = PredictStub()
    documents = [
        RetrievedChunk(chunk_id="wrong", content="irrelevant", score=0.99, stage1_rank=1),
        RetrievedChunk(chunk_id="correct", content="highly relevant", score=0.45, stage1_rank=2),
    ]
    reordered = reranker.rerank("the query", documents, k2=2)
    assert [item.chunk_id for item in reordered] == ["correct", "wrong"]
    assert reordered[0].score == pytest.approx(0.9820138, abs=1e-5)
    assert reordered[0].raw_logit == 4.0
    assert reordered[0].stage1_rank == 2
    assert reordered[0].stage2_rank == 1


def test_reranker_with_empty_candidates_does_not_load_model():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    assert reranker.rerank("anything", []) == []
