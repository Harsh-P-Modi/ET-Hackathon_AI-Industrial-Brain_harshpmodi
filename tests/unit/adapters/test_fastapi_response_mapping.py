# Feature: fastapi-adapter, Property 2: Response mapping round-trip
"""Property-based test: response mapping round-trip.

Validates: Requirements 2.2, 2.3

For any valid SynthesizedResponse (with arbitrary answer_text, any RouteType value,
and any tuple of Citation objects), the JSON response from /query SHALL contain:
- `answer` equal to `answer_text`
- `route_used` equal to `route_used.value`
- `citations` array with one object per Citation containing matching chunk_id,
  source_document, and snippet.
"""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter
from src.domain.entities import Citation, RouteType, SynthesizedResponse


# ─── Hypothesis Strategies ────────────────────────────────────────────────────

citation_strategy = st.builds(
    Citation,
    chunk_id=st.text(min_size=1),
    source_document=st.text(min_size=1),
    snippet=st.text(min_size=1),
)

synthesized_response_strategy = st.builds(
    SynthesizedResponse,
    answer_text=st.text(min_size=1),
    citations=st.lists(citation_strategy, min_size=0, max_size=10).map(tuple),
    route_used=st.sampled_from(RouteType),
)


# ─── Property Test ────────────────────────────────────────────────────────────


@given(domain_response=synthesized_response_strategy)
@settings(max_examples=100)
def test_response_mapping_round_trip(domain_response: SynthesizedResponse) -> None:
    """For any SynthesizedResponse, JSON fields match domain values exactly."""
    mock_query = Mock()
    mock_query.ask.return_value = domain_response

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

    resp = client.post("/query", json={"question": "test question"})

    assert resp.status_code == 200
    body = resp.json()

    # answer maps from answer_text
    assert body["answer"] == domain_response.answer_text

    # route_used maps from route_used.value (the enum string value)
    assert body["route_used"] == domain_response.route_used.value

    # citations array has same length
    assert len(body["citations"]) == len(domain_response.citations)

    # each citation maps field-for-field
    for json_citation, domain_citation in zip(
        body["citations"], domain_response.citations
    ):
        assert json_citation["chunk_id"] == domain_citation.chunk_id
        assert json_citation["source_document"] == domain_citation.source_document
        assert json_citation["snippet"] == domain_citation.snippet
