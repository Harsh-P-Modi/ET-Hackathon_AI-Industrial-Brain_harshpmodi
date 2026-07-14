"""Unit tests for Pydantic validation via HTTP endpoints.

Validates Requirements 2.5, 2.6, 3.3 — input validation returns HTTP 422
when request bodies are invalid.
"""

import pytest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter
from src.domain.entities import Citation, RouteType, SynthesizedResponse


@pytest.fixture
def client():
    """Create a TestClient with mock dependencies."""
    mock_query = Mock()
    mock_query.ask.return_value = SynthesizedResponse(
        answer_text="test answer",
        citations=(),
        route_used=RouteType.VECTOR,
    )
    mock_ingestion = Mock()
    mock_vector = Mock()
    mock_graph = Mock()
    mock_llm = Mock()

    adapter = FastAPIAdapter(
        query_service=mock_query,
        ingestion_service=mock_ingestion,
        vector_store=mock_vector,
        graph_store=mock_graph,
        llm_inference=mock_llm,
        neo4j_ping=lambda: True,
        ollama_ping=lambda: True,
    )
    return TestClient(adapter.app)


class TestQueryValidation:
    """POST /query input validation — Requirement 2.5, 2.6."""

    def test_empty_string_returns_422(self, client):
        """Empty string question is rejected with HTTP 422."""
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_whitespace_only_returns_422(self, client):
        """Whitespace-only question is rejected with HTTP 422."""
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 422

    def test_valid_question_returns_200(self, client):
        """A valid non-empty question returns HTTP 200."""
        resp = client.post("/query", json={"question": "What is the pump flow rate?"})
        assert resp.status_code == 200

    def test_missing_question_field_returns_422(self, client):
        """Missing question field returns HTTP 422."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_tabs_newlines_only_returns_422(self, client):
        """Tabs and newlines only is rejected with HTTP 422."""
        resp = client.post("/query", json={"question": "\t\n\r"})
        assert resp.status_code == 422


class TestIngestValidation:
    """POST /ingest input validation — Requirement 3.3."""

    def test_missing_file_returns_422(self, client):
        """Missing file in multipart upload returns HTTP 422."""
        resp = client.post("/ingest")
        assert resp.status_code == 422
