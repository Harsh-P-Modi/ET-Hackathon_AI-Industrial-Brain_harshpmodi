"""Unit tests for the embed method with SHA-256 cache (task 4.1).

Validates:
- Cache hit returns stored result without HTTP call
- Cache miss calls Ollama /api/embeddings with correct model
- Dimension validation raises ValueError on mismatch
- OllamaAdapterError raised on network errors
- OllamaAdapterError raised on missing 'embedding' field
"""

import os

os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("ENVIRONMENT", "test")

from unittest.mock import patch, MagicMock

import pytest

from src.config import VECTOR_DIMENSIONS


def _make_adapter():
    """Create a LangGraphOrchestratorAdapter with mocked health check."""
    with patch("src.adapters.outbound.ollama_llm_adapter.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter
        from src.domain.services import HybridContextFuser

        fake_vector = MagicMock()
        fake_graph = MagicMock()
        fake_fuser = MagicMock(spec=HybridContextFuser)

        adapter = LangGraphOrchestratorAdapter(
            vector_store=fake_vector,
            graph_store=fake_graph,
            fuser=fake_fuser,
        )
    return adapter


class TestEmbedCacheHit:
    """Cache hits should return immediately without HTTP call."""

    def test_cache_hit_returns_stored_embedding(self):
        adapter = _make_adapter()
        # Pre-populate cache.
        import hashlib
        text = "test embedding text"
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        expected = [0.1] * VECTOR_DIMENSIONS
        adapter._embedding_cache[cache_key] = expected

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post") as mock_post:
            result = adapter.embed(text)
            mock_post.assert_not_called()

        assert result == expected

    def test_cache_prevents_duplicate_http_calls(self):
        adapter = _make_adapter()
        embedding = [0.5] * VECTOR_DIMENSIONS

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response) as mock_post:
            result1 = adapter.embed("hello world")
            result2 = adapter.embed("hello world")

        assert result1 == result2
        assert mock_post.call_count == 1


class TestEmbedCacheMiss:
    """Cache misses should call Ollama and store result."""

    def test_calls_correct_endpoint_and_model(self):
        adapter = _make_adapter()
        embedding = [0.2] * VECTOR_DIMENSIONS

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response) as mock_post:
            adapter.embed("some text")

        call_args = mock_post.call_args
        assert "/api/embeddings" in call_args[0][0] or "/api/embeddings" in call_args.kwargs.get("url", call_args[0][0] if call_args[0] else "")
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "nomic-embed-text"
        assert payload["prompt"] == "some text"

    def test_returns_embedding_and_caches_it(self):
        adapter = _make_adapter()
        embedding = [0.3] * VECTOR_DIMENSIONS

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response):
            result = adapter.embed("cache me")

        assert result == embedding
        # Verify it's now in the cache.
        import hashlib
        cache_key = hashlib.sha256("cache me".encode()).hexdigest()
        assert cache_key in adapter._embedding_cache


class TestEmbedDimensionValidation:
    """Dimension mismatch should raise ValueError."""

    def test_wrong_dimension_raises_value_error(self):
        adapter = _make_adapter()
        # Return a 512-dim embedding instead of 768.
        wrong_embedding = [0.1] * 512

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": wrong_embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response):
            with pytest.raises(ValueError, match=r"expected 768.*got 512"):
                adapter.embed("bad dims")

    def test_empty_embedding_raises_value_error(self):
        adapter = _make_adapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": []}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response):
            with pytest.raises(ValueError, match=r"expected 768.*got 0"):
                adapter.embed("empty response")


class TestEmbedNetworkErrors:
    """Network/HTTP errors should raise OllamaAdapterError."""

    def test_connection_error_raises_adapter_error(self):
        import requests as req
        from src.adapters.outbound.ollama_llm_adapter import OllamaAdapterError

        adapter = _make_adapter()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", side_effect=req.ConnectionError("refused")):
            with pytest.raises(OllamaAdapterError) as exc_info:
                adapter.embed("connection fail")

        assert exc_info.value.operation == "embed"
        assert "/api/embeddings" in exc_info.value.endpoint
        assert "refused" in exc_info.value.detail

    def test_timeout_raises_adapter_error(self):
        import requests as req
        from src.adapters.outbound.ollama_llm_adapter import OllamaAdapterError

        adapter = _make_adapter()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", side_effect=req.Timeout("timed out")):
            with pytest.raises(OllamaAdapterError) as exc_info:
                adapter.embed("timeout text")

        assert exc_info.value.operation == "embed"
        assert "timed out" in exc_info.value.detail

    def test_missing_embedding_field_raises_adapter_error(self):
        from src.adapters.outbound.ollama_llm_adapter import OllamaAdapterError

        adapter = _make_adapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "model not found"}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response):
            with pytest.raises(OllamaAdapterError) as exc_info:
                adapter.embed("no embedding key")

        assert exc_info.value.operation == "embed"
        assert "embedding" in exc_info.value.detail.lower()
