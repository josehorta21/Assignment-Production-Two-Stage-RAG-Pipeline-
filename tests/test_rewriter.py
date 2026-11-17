import pytest
from src.rewriter import QueryRewriter


def test_pronoun_resolution_expands_prior_topic():
    rewritten = QueryRewriter().rewrite("What are its system requirements?", ["Kubernetes Worker Nodes"])
    assert "Kubernetes Worker Nodes" in rewritten
    assert "system requirements" in rewritten


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="blank"):
        QueryRewriter().rewrite("  ")


def test_preserves_exact_configuration_identifier():
    query = "What does CUSTOM_CONFIG_FLAG_XYZ do during maxUnavailable rollout?"
    assert "CUSTOM_CONFIG_FLAG_XYZ" in QueryRewriter().rewrite(query)
    assert "maxUnavailable" in QueryRewriter().rewrite(query)


def test_local_llm_timeout_falls_back():
    class BrokenLocalModel:
        def generate(self, instruction, max_new_tokens=100):
            raise TimeoutError("Local model stalled")
    result = QueryRewriter(llm=BrokenLocalModel(), mode="local").rewrite("What is maxUnavailable?", [])
    assert "maxUnavailable" in result


def test_local_llm_cannot_delete_identifier():
    class UnsafeRewriter:
        def generate(self, instruction, max_new_tokens=100):
            return "configuration flags"
    result = QueryRewriter(llm=UnsafeRewriter(), mode="local").rewrite("What is CUSTOM_CONFIG_FLAG_XYZ?", [])
    assert "CUSTOM_CONFIG_FLAG_XYZ" in result
