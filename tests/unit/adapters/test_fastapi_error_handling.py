"""Unit tests for error handling in FastAPI adapter endpoints.

Validates:
- /query returns 500 with generic message when domain raises (Req 5.1)
- Response body does not contain exception message or stack trace (Req 5.2)
- Exception is logged server-side (Req 5.3)
"""

import logging

import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter


@pytest.fixture
def failing_client():
    """Create TestClient where query_service.ask() raises with sensitive info."""
    mock_query = Mock()
    mock_query.ask.side_effect = RuntimeError(
        "neo4j://secret-host:7687 connection refused"
    )

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


class TestQueryErrorHandling:
    """Tests for /query error handling behavior."""

    def test_returns_500_on_domain_exception(self, failing_client):
        """POST /query returns HTTP 500 when the domain service raises."""
        resp = failing_client.post("/query", json={"question": "What is X?"})
        assert resp.status_code == 500

    def test_returns_generic_error_message(self, failing_client):
        """POST /query returns generic error detail, not exception specifics."""
        resp = failing_client.post("/query", json={"question": "What is X?"})
        body = resp.json()
        assert body == {"detail": "Internal server error"}

    def test_does_not_leak_exception_message(self, failing_client):
        """Response body must not contain connection strings, error text, or tracebacks."""
        resp = failing_client.post("/query", json={"question": "What is X?"})
        body_text = resp.text
        assert "neo4j://secret-host:7687" not in body_text
        assert "connection refused" not in body_text
        assert "RuntimeError" not in body_text
        assert "Traceback" not in body_text

    def test_exception_is_logged_server_side(self, failing_client, caplog):
        """The full exception detail is logged server-side for debugging."""
        with caplog.at_level(
            logging.ERROR, logger="src.adapters.inbound.fastapi_adapter"
        ):
            failing_client.post("/query", json={"question": "What is X?"})

        assert len(caplog.records) >= 1
        # The log record should contain the original exception info
        log_record = caplog.records[0]
        assert "neo4j://secret-host:7687" in log_record.message or (
            log_record.exc_info is not None
            and "connection refused" in str(log_record.exc_info[1])
        )
