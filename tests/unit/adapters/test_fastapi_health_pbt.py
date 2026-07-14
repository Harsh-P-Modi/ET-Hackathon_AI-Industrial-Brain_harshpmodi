# Feature: fastapi-adapter, Property 5: Health check status reflection
"""Property-based test: Health check status reflection.

For any combination of (neo4j_healthy: bool, ollama_healthy: bool),
the /health endpoint returns HTTP 200 with neo4j and ollama fields
matching the respective backend availability.

**Validates: Requirements 4.3, 4.4, 4.5, 4.6**
"""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter


@given(neo4j_healthy=st.booleans(), ollama_healthy=st.booleans())
@settings(max_examples=100)
def test_health_reflects_backend_status(neo4j_healthy: bool, ollama_healthy: bool):
    """Health endpoint reflects exact backend status for all bool combinations."""
    adapter = FastAPIAdapter(
        query_service=Mock(),
        ingestion_service=Mock(),
        vector_store=Mock(),
        graph_store=Mock(),
        llm_inference=Mock(),
        neo4j_ping=lambda: neo4j_healthy,
        ollama_ping=lambda: ollama_healthy,
    )
    client = TestClient(adapter.app)

    resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["neo4j"] == neo4j_healthy
    assert data["ollama"] == ollama_healthy
    assert data["status"] == "ok"
