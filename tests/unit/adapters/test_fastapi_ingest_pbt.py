# Feature: fastapi-adapter, Property 4: Ingest delegation and response
"""Property-based test verifying ingest endpoint delegates correctly and returns expected response.

Uses Hypothesis to verify that for any byte sequence and non-empty filename,
the adapter calls DocumentIngestionPort.ingest() with the exact values and
returns the correct response shape.

**Validates: Requirements 3.1, 3.2**
"""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter


# Strategies
random_bytes = st.binary(min_size=1)
# Use filename-safe ASCII characters that won't be percent-encoded in multipart
# Content-Disposition headers by httpx/starlette TestClient.
_FILENAME_SAFE_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_.() "
)
random_filenames = st.text(
    alphabet=_FILENAME_SAFE_CHARS,
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip())


def _make_client(mock_ingestion: Mock) -> TestClient:
    """Build a TestClient with the given mock ingestion service."""
    adapter = FastAPIAdapter(
        query_service=Mock(),
        ingestion_service=mock_ingestion,
        vector_store=Mock(),
        graph_store=Mock(),
        llm_inference=Mock(),
        neo4j_ping=lambda: True,
        ollama_ping=lambda: True,
    )
    return TestClient(adapter.app)


class TestIngestDelegationAndResponse:
    """Property 4: Ingest delegation and response.

    **Validates: Requirements 3.1, 3.2**

    For any byte sequence and any non-empty filename, uploading via multipart
    to /ingest must call DocumentIngestionPort.ingest() with the exact bytes
    and filename, and return {"status": "accepted", "filename": <filename>}.
    """

    @given(content=random_bytes, filename=random_filenames)
    @settings(max_examples=100)
    def test_ingest_called_with_exact_values_and_response_matches(
        self, content: bytes, filename: str
    ) -> None:
        """The adapter delegates exact bytes/filename and returns accepted response."""
        mock_ingestion = Mock()
        client = _make_client(mock_ingestion)

        response = client.post(
            "/ingest",
            files={"file": (filename, content, "application/octet-stream")},
        )

        assert response.status_code == 200
        mock_ingestion.ingest.assert_called_once_with(content, filename)

        body = response.json()
        assert body["status"] == "accepted"
        assert body["filename"] == filename
