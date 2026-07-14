# Feature: fastapi-adapter, Property 1: Query delegation preserves input
"""Property-based tests for the FastAPI adapter.

Uses Hypothesis to verify universal properties of the adapter's HTTP interface.
"""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter
from src.domain.entities import Citation, RouteType, SynthesizedResponse


def _make_client(mock_query: Mock) -> TestClient:
    """Build a TestClient with the given mock query service."""
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


class TestQueryDelegationPreservesInput:
    """Property 1: Query delegation preserves input.

    **Validates: Requirements 2.1**

    For any non-whitespace question string, POSTing to /query must call the
    domain query service's ask() method with the exact same string.
    """

    @given(question=st.text(min_size=1).filter(lambda s: s.strip()))
    @settings(max_examples=100)
    def test_ask_called_with_exact_question(self, question: str) -> None:
        """The adapter delegates the exact question string to ask()."""
        mock_query = Mock()
        mock_query.ask.return_value = SynthesizedResponse(
            answer_text="stub answer",
            citations=(
                Citation(
                    chunk_id="c1",
                    source_document="doc.pdf",
                    snippet="snippet text",
                ),
            ),
            route_used=RouteType.VECTOR,
        )

        client = _make_client(mock_query)
        response = client.post("/query", json={"question": question})

        assert response.status_code == 200
        mock_query.ask.assert_called_once_with(question)
