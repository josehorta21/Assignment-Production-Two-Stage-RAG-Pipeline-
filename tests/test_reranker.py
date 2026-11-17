import math
import pytest
from src.schemas import RetrievedChunk
from src.reranker import sigmoid, route_confidence
from src.calibration import choose_threshold


def test_sigmoid_handles_positive_negative_logits():
    assert sigmoid(-3) == pytest.approx(0.04742587, abs=1e-6)
    assert sigmoid(0) == 0.5
    assert sigmoid(3) == pytest.approx(0.95257413, abs=1e-6)
    assert 0 <= sigmoid(-1e6) < sigmoid(-3)
    assert sigmoid(1e6) == 1.0


def test_threshold_gate_matches_two_mock_scores():
    poor = RetrievedChunk(chunk_id="low", content="x", score=0.42)
    good = RetrievedChunk(chunk_id="high", content="x", score=0.81)
    assert route_confidence([poor], 0.55) == (0.42, False)
    assert route_confidence([good], 0.55) == (0.81, True)


def test_empty_candidates_force_web():
    assert route_confidence([], 0.55) == (0.0, False)


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        route_confidence([], 1.2)


def test_threshold_calibration_penalizes_false_internal_more():
    observed = [
        {"score": 0.8, "expected": "INTERNAL_RAG"},
        {"score": 0.7, "expected": "WEB_FALLBACK"},
        {"score": 0.3, "expected": "WEB_FALLBACK"},
    ]
    chosen = choose_threshold(observed, [0.55, 0.75])
    assert chosen["selected"]["threshold"] == 0.75
    assert chosen["selected"]["false_internal_count"] == 0
