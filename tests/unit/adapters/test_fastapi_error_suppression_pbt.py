# Feature: fastapi-adapter, Property 6: Error suppression on domain failure
"""Property-based test: Error suppression on domain failure.

For any exception raised by KnowledgeQueryPort.ask() (regardless of exception
type or message content), the /query endpoint returns HTTP 500 with body
{"detail": "Internal server error"}, and the response body does NOT contain
the exception message, stack trace, or any substring from the original error.

**Validates: Requirements 5.1, 5.2**
"""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter


# Generate random error messages, including strings that look like connection strings
error_messages = st.one_of(
    st.text(min_size=1),
    st.from_regex(r"neo4j://[a-z]+:[a-z]+@[a-z]+:\d+", fullmatch=True),
    st.from_regex(r"http://[a-z]+:\d+/api/[a-z]+", fullmatch=True),
)


@given(error_msg=error_messages)
@settings(max_examples=100)
def test_error_suppression(error_msg: str):
    """Domain exceptions never leak into the HTTP response body."""
    mock_query = Mock()
    mock_query.ask.side_effect = RuntimeError(error_msg)

    adapter = FastAPIAdapter(
        query_service=mock_query,
        ingestion_service=Mock(),
        vector_store=Mock(),
        graph_store=Mock(),
        llm_inference=Mock(),
        neo4j_ping=lambda: True,
        ollama_ping=lambda: True,
    )
    client = TestClient(adapter.app, raise_server_exceptions=False)

    resp = client.post("/query", json={"question": "test question"})

    assert resp.status_code == 500
    body = resp.json()
    assert body == {"detail": "Internal server error"}
    # The error message must not appear in the response body
    if error_msg.strip():  # Only check non-empty messages
        assert error_msg not in resp.text
