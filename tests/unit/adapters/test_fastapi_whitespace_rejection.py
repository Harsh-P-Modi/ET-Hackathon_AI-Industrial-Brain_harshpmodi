# Feature: fastapi-adapter, Property 3: Whitespace-only queries are rejected
"""Property-based test: whitespace-only queries are rejected with HTTP 422.

**Validates: Requirements 2.5, 2.6**

Generates whitespace-only strings (including empty string) and verifies that
the POST /query endpoint rejects them with HTTP 422 and never invokes the
domain query service.
"""

import string
from unittest.mock import Mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from fastapi.testclient import TestClient

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter


@pytest.fixture
def mock_query():
    """Create a mock query service."""
    return Mock()


@pytest.fixture
def client(mock_query):
    """Create a TestClient with mock dependencies."""
    adapter = FastAPIAdapter(
        query_service=mock_query,
        ingestion_service=Mock(),
        vector_store=Mock(),
        graph_store=Mock(),
        llm_inference=Mock(),
        neo4j_ping=lambda: True,
        ollama_ping=lambda: True,
    )
    return TestClient(adapter.app)


@given(question=st.text(alphabet=string.whitespace))
@settings(max_examples=100)
def test_whitespace_only_queries_rejected(question: str):
    """For any whitespace-only string (including empty), POST /query returns 422
    and the domain ask() method is never called.

    **Validates: Requirements 2.5, 2.6**
    """
    mock_query = Mock()
    adapter = FastAPIAdapter(
        query_service=mock_query,
        ingestion_service=Mock(),
        vector_store=Mock(),
        graph_store=Mock(),
        llm_inference=Mock(),
        neo4j_ping=lambda: True,
        ollama_ping=lambda: True,
    )
    client = TestClient(adapter.app)

    resp = client.post("/query", json={"question": question})

    assert resp.status_code == 422, (
        f"Expected 422 for whitespace-only input {question!r}, got {resp.status_code}"
    )
    mock_query.ask.assert_not_called()
