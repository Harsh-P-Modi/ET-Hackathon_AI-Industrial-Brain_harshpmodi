"""Unit tests for health check and error handling (task 8.3).

Validates:
- Health check failure raises RuntimeError at adapter init (Req 1.4, 1.5)
- OllamaAdapterError exceptions contain operation, endpoint, and error detail (Req 6.2, 6.4)
"""

import os

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("NEO4J_PASSWORD", "test")

from unittest.mock import patch, MagicMock

import pytest
import requests as req

from src.adapters.outbound.ollama_llm_adapter import OllamaAdapterError


def _make_adapter():
    """Create a LangGraphOrchestratorAdapter with mocked healthy health check."""
    with patch("src.adapters.outbound.ollama_llm_adapter.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter
        from src.domain.services import HybridContextFuser

        adapter = LangGraphOrchestratorAdapter(
            vector_store=MagicMock(),
            graph_store=MagicMock(),
            fuser=MagicMock(spec=HybridContextFuser),
        )
    return adapter


class TestHealthCheckFailureRaises:
    """Unreachable Ollama daemon should raise RuntimeError at init."""

    def test_health_check_failure_raises(self):
        """ConnectionError during health check raises RuntimeError with 'not reachable'."""
        from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter
        from src.domain.services import HybridContextFuser

        with patch(
            "src.adapters.outbound.ollama_llm_adapter.requests.get",
            side_effect=req.ConnectionError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match=r"not reachable"):
                LangGraphOrchestratorAdapter(
                    vector_store=MagicMock(),
                    graph_store=MagicMock(),
                    fuser=MagicMock(spec=HybridContextFuser),
                )

    def test_health_check_timeout_raises(self):
        """Timeout during health check raises RuntimeError with host URL."""
        from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter
        from src.domain.services import HybridContextFuser

        with patch(
            "src.adapters.outbound.ollama_llm_adapter.requests.get",
            side_effect=req.Timeout("timed out"),
        ):
            with pytest.raises(RuntimeError, match=r"localhost:11434"):
                LangGraphOrchestratorAdapter(
                    vector_store=MagicMock(),
                    graph_store=MagicMock(),
                    fuser=MagicMock(spec=HybridContextFuser),
                )


class TestErrorMessagesContainDetails:
    """OllamaAdapterError exceptions include operation, endpoint, and error detail."""

    def test_embed_error_contains_details(self):
        """embed() ConnectionError produces OllamaAdapterError with structured fields."""
        adapter = _make_adapter()

        with patch(
            "src.adapters.outbound.ollama_llm_adapter.requests.post",
            side_effect=req.ConnectionError("host unreachable"),
        ):
            with pytest.raises(OllamaAdapterError) as exc_info:
                adapter.embed("test text")

        err = exc_info.value
        assert err.operation == "embed"
        assert "/api/embeddings" in err.endpoint
        assert "host unreachable" in err.detail

    def test_synthesize_error_contains_details(self):
        """synthesize() ConnectionError produces OllamaAdapterError with structured fields."""
        adapter = _make_adapter()

        with patch(
            "src.adapters.outbound.ollama_llm_adapter.requests.post",
            side_effect=req.ConnectionError("connection reset"),
        ):
            with pytest.raises(OllamaAdapterError) as exc_info:
                adapter.synthesize("What is X?", [])

        err = exc_info.value
        assert err.operation == "synthesize"
        assert "/api/generate" in err.endpoint
        assert "connection reset" in err.detail

    def test_error_string_representation_includes_all_fields(self):
        """The string representation of OllamaAdapterError includes all diagnostic info."""
        error = OllamaAdapterError(
            operation="embed",
            endpoint="http://localhost:11434/api/embeddings",
            detail="Connection refused",
        )
        error_str = str(error)
        assert "embed" in error_str
        assert "/api/embeddings" in error_str
        assert "Connection refused" in error_str
