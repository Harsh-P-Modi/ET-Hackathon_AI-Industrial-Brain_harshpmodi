"""Unit tests for Pydantic request/response models in fastapi_adapter.py."""

import pytest
from pydantic import ValidationError

from src.adapters.inbound.fastapi_adapter import (
    CitationResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)


class TestQueryRequest:
    """Tests for QueryRequest validation."""

    def test_valid_question_accepted(self):
        req = QueryRequest(question="What is the pump flow rate?")
        assert req.question == "What is the pump flow rate?"

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="   ")

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_missing_question_field_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest()  # type: ignore[call-arg]

    def test_tabs_and_newlines_only_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="\t\n  \r")


class TestCitationResponse:
    """Tests for CitationResponse model."""

    def test_all_fields_set(self):
        c = CitationResponse(
            chunk_id="c1", source_document="manual.pdf", snippet="check valve"
        )
        assert c.chunk_id == "c1"
        assert c.source_document == "manual.pdf"
        assert c.snippet == "check valve"


class TestQueryResponse:
    """Tests for QueryResponse model."""

    def test_with_citations(self):
        citation = CitationResponse(
            chunk_id="c1", source_document="doc.pdf", snippet="text"
        )
        resp = QueryResponse(
            answer="The answer", route_used="VECTOR", citations=[citation]
        )
        assert resp.answer == "The answer"
        assert resp.route_used == "VECTOR"
        assert len(resp.citations) == 1

    def test_with_empty_citations(self):
        resp = QueryResponse(answer="No data", route_used="GRAPH_LOCAL", citations=[])
        assert resp.citations == []


class TestIngestResponse:
    """Tests for IngestResponse model."""

    def test_default_status(self):
        resp = IngestResponse(filename="doc.pdf")
        assert resp.status == "accepted"
        assert resp.filename == "doc.pdf"

    def test_custom_status(self):
        resp = IngestResponse(status="processing", filename="report.pdf")
        assert resp.status == "processing"


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_default_status(self):
        resp = HealthResponse(neo4j=True, ollama=False)
        assert resp.status == "ok"
        assert resp.neo4j is True
        assert resp.ollama is False


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_default_detail(self):
        resp = ErrorResponse()
        assert resp.detail == "Internal server error"

    def test_custom_detail(self):
        resp = ErrorResponse(detail="Not found")
        assert resp.detail == "Not found"
