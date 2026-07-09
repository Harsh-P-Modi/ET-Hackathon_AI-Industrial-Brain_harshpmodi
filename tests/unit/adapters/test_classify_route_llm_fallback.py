"""Unit tests for classify_route LLM fallback logic (task 3.2).

These tests validate that when no keyword prefilter matches, the adapter
invokes Ollama /api/generate with JSON mode and correctly handles various
response scenarios: valid routes, invalid JSON, invalid routes, and network errors.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
import requests


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


class TestClassifyRouteLLMFallbackSuccess:
    """When LLM returns a valid route, classify_route should return it."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = _make_adapter()

    @pytest.mark.parametrize(
        "route",
        [
            "vector_search",
            "graph_local_search",
            "graph_global_search",
            "hybrid_fusion",
        ],
    )
    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_valid_route_from_llm(self, mock_post, route):
        """LLM returns a valid route value — should be returned directly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({"route": route})
        }
        mock_post.return_value = mock_response

        result = self.adapter.classify_route("How do I replace a bearing?")
        assert result == route

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_llm_called_with_json_mode(self, mock_post):
        """When no keywords match, Ollama is called with format: 'json' and stream: false."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({"route": "vector_search"})
        }
        mock_post.return_value = mock_response

        self.adapter.classify_route("What is vibration analysis?")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["format"] == "json"
        assert payload["stream"] is False
        assert payload["model"] == self.adapter._model
        assert payload["prompt"] == "What is vibration analysis?"
        assert "system" in payload

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_llm_not_called_when_keyword_matches(self, mock_post):
        """Keywords match should NOT trigger any HTTP POST to Ollama."""
        self.adapter.classify_route("What is downstream of P-101?")
        mock_post.assert_not_called()


class TestClassifyRouteLLMFallbackErrors:
    """Error scenarios should all return 'hybrid_fusion'."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = _make_adapter()

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_network_connection_error_returns_hybrid(self, mock_post):
        """ConnectionError from requests should return hybrid_fusion."""
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        result = self.adapter.classify_route("How do I replace a bearing?")
        assert result == "hybrid_fusion"

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_timeout_error_returns_hybrid(self, mock_post):
        """Timeout from requests should return hybrid_fusion."""
        mock_post.side_effect = requests.Timeout("Request timed out")

        result = self.adapter.classify_route("What is vibration analysis?")
        assert result == "hybrid_fusion"

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_http_error_returns_hybrid(self, mock_post):
        """HTTP error (e.g., 500) should return hybrid_fusion."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        result = self.adapter.classify_route("Tell me about maintenance")
        assert result == "hybrid_fusion"

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_invalid_outer_json_returns_hybrid(self, mock_post):
        """If response.json() fails, should return hybrid_fusion."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("No JSON in response")
        mock_post.return_value = mock_response

        result = self.adapter.classify_route("How do I replace a bearing?")
        assert result == "hybrid_fusion"

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_invalid_inner_json_returns_hybrid(self, mock_post):
        """If inner 'response' field is not valid JSON, should return hybrid_fusion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "not valid json at all"}
        mock_post.return_value = mock_response

        result = self.adapter.classify_route("How do I replace a bearing?")
        assert result == "hybrid_fusion"

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_missing_route_key_returns_hybrid(self, mock_post):
        """If inner JSON has no 'route' key, should return hybrid_fusion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps({"classification": "vector_search"})
        }
        mock_post.return_value = mock_response

        result = self.adapter.classify_route("How do I replace a bearing?")
        assert result == "hybrid_fusion"

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_invalid_route_value_returns_hybrid(self, mock_post):
        """If route value is not a valid RouteType, should return hybrid_fusion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps({"route": "invalid_route_type"})
        }
        mock_post.return_value = mock_response

        result = self.adapter.classify_route("How do I replace a bearing?")
        assert result == "hybrid_fusion"

    @patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
    def test_missing_response_field_returns_hybrid(self, mock_post):
        """If outer JSON has no 'response' field, should return hybrid_fusion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"model": "qwen2.5:7b-instruct"}
        mock_post.return_value = mock_response

        result = self.adapter.classify_route("How do I replace a bearing?")
        assert result == "hybrid_fusion"
