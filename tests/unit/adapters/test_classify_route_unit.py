"""Unit tests for classify_route with exact task 8.1 function names.

These tests verify the 5 specific scenarios required by task 8.1:
keyword routing (GRAPH_LOCAL, GRAPH_GLOBAL), LLM fallback with JSON mode,
malformed JSON fallback, and network error fallback.
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


@patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
def test_classify_route_keyword_graph_local(mock_post):
    """Keyword hit returns GRAPH_LOCAL, no HTTP call made."""
    adapter = _make_adapter()

    result = adapter.classify_route("What is downstream of pump P-101?")

    assert result == "graph_local_search"
    mock_post.assert_not_called()


@patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
def test_classify_route_keyword_graph_global(mock_post):
    """Keyword hit returns GRAPH_GLOBAL, no HTTP call made."""
    adapter = _make_adapter()

    result = adapter.classify_route("What are common causes of bearing failure?")

    assert result == "graph_global_search"
    mock_post.assert_not_called()


@patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
def test_classify_route_llm_fallback(mock_post):
    """No keywords triggers Ollama call with JSON mode."""
    adapter = _make_adapter()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps({"route": "vector_search"})
    }
    mock_post.return_value = mock_response

    result = adapter.classify_route("How do I replace a bearing?")

    assert result == "vector_search"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
    assert payload["format"] == "json"
    assert payload["stream"] is False


@patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
def test_classify_route_invalid_json_fallback(mock_post):
    """Malformed response returns HYBRID."""
    adapter = _make_adapter()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "not valid json at all"}
    mock_post.return_value = mock_response

    result = adapter.classify_route("How do I replace a bearing?")

    assert result == "hybrid_fusion"


@patch("src.adapters.outbound.ollama_llm_adapter.requests.post")
def test_classify_route_network_error_fallback(mock_post):
    """Connection error returns HYBRID."""
    adapter = _make_adapter()

    mock_post.side_effect = requests.ConnectionError("Connection refused")

    result = adapter.classify_route("How do I replace a bearing?")

    assert result == "hybrid_fusion"
