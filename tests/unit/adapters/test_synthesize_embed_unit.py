"""Unit tests for synthesize and embed methods (task 8.2).

Validates:
- Requirements 3.2, 3.3: synthesize citation instruction and grounded-only rule
- Requirements 4.2, 4.3, 4.4: embed model, dimensionality, and cache behavior
"""

import os

os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("ENVIRONMENT", "test")

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from src.config import VECTOR_DIMENSIONS
from src.domain.entities import DocumentChunk, EquipmentNode, MaintenanceEvent


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


def _make_chunk(chunk_id: str, text: str = "some text") -> DocumentChunk:
    """Helper to create a DocumentChunk for testing."""
    return DocumentChunk(
        chunk_id=chunk_id,
        parent_id="parent_1",
        text=text,
        embedding=None,
        source_document="manual.pdf",
    )


class TestSynthesizeIncludesCitationsInstruction:
    """When 2+ context items are provided, system prompt includes citation instruction.

    Validates: Requirements 3.2, 3.3
    """

    def test_synthesize_includes_citations_instruction(self):
        """2+ context items triggers citation instruction in system prompt."""
        adapter = _make_adapter()

        # Create 2 context items to trigger citation instruction.
        context = [
            _make_chunk("c1", "Pump P-100 requires oil change every 500 hours."),
            _make_chunk("c2", "Filter F-200 should be replaced monthly."),
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "The pump needs oil change."}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response) as mock_post:
            adapter.synthesize("What maintenance is needed?", context)

        # Capture the POST payload's system field.
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        system_prompt = payload["system"]

        assert "Cite sources inline" in system_prompt


class TestSynthesizeContextOnlyInstruction:
    """System prompt always contains grounded-only rule.

    Validates: Requirements 3.2, 3.3
    """

    def test_synthesize_context_only_instruction(self):
        """System prompt contains grounded-only rule regardless of context count."""
        adapter = _make_adapter()

        # Use only 1 context item (no citation instruction expected).
        context = [
            _make_chunk("c1", "Valve V-300 operating temperature is 150C."),
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "The valve operates at 150C."}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response) as mock_post:
            adapter.synthesize("What is the operating temp?", context)

        # Capture the POST payload's system field.
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        system_prompt = payload["system"]

        assert "Answer ONLY from the provided context" in system_prompt


class TestEmbedCallsCorrectModel:
    """Embed request payload uses OLLAMA_EMBED_MODEL.

    Validates: Requirements 4.2
    """

    def test_embed_calls_correct_model(self):
        """Request payload model field matches 'nomic-embed-text'."""
        adapter = _make_adapter()
        embedding = [0.1] * VECTOR_DIMENSIONS

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response) as mock_post:
            adapter.embed("test text")

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "nomic-embed-text"


class TestEmbedValidatesDimensionality:
    """Non-768 response raises ValueError.

    Validates: Requirements 4.3
    """

    def test_embed_validates_dimensionality(self):
        """512-dimension response raises ValueError."""
        adapter = _make_adapter()
        wrong_embedding = [0.1] * 512

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": wrong_embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response):
            with pytest.raises(ValueError, match=r"expected 768.*got 512"):
                adapter.embed("bad dimensions text")


class TestEmbedCachePreventsRecompute:
    """Same text twice results in only one HTTP call.

    Validates: Requirements 4.4
    """

    def test_embed_cache_prevents_recompute(self):
        """Calling embed with same text twice only makes one HTTP call."""
        adapter = _make_adapter()
        embedding = [0.5] * VECTOR_DIMENSIONS

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("src.adapters.outbound.ollama_llm_adapter.requests.post", return_value=mock_response) as mock_post:
            result1 = adapter.embed("same")
            result2 = adapter.embed("same")

        assert result1 == result2
        assert mock_post.call_count == 1
