"""True pairwise cross-encoder ranking and stable logit normalization."""
from __future__ import annotations

import math
from .schemas import RetrievedChunk


def sigmoid(logit: float) -> float:
    # Numerically stable for large positive/negative logits.
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def route_confidence(candidates: list[RetrievedChunk], threshold: float) -> tuple[float, bool]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must be in [0, 1]")
    top = max((x.score for x in candidates), default=0.0)
    return top, bool(candidates) and top >= threshold


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        import torch
        self.model = CrossEncoder(model_name, activation_fn=torch.nn.Identity())

    def rerank(self, query: str, candidates: list[RetrievedChunk], k2: int = 5) -> list[RetrievedChunk]:
        if not candidates:
            return []
        import torch
        pairs = [(query, passage.content[:1400]) for passage in candidates]
        # Explicit Identity is essential: we compute sigmoid exactly once.
        raw = self.model.predict(pairs, activation_fn=torch.nn.Identity(),
                                 show_progress_bar=False, batch_size=16)
        ranked = []
        for passage, logit in zip(candidates, raw):
            copied = passage.model_copy(deep=True)
            copied.raw_logit = float(logit)
            copied.score = sigmoid(float(logit))
            ranked.append(copied)
        ranked.sort(key=lambda item: (-item.score, item.chunk_id))
        for rank, item in enumerate(ranked, 1):
            item.stage2_rank = rank
        return ranked[:max(k2, 0)]
