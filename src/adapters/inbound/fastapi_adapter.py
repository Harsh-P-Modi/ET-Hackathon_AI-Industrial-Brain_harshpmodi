"""FastAPI inbound adapter — Pydantic request/response models and HTTP handlers."""

import logging
from collections.abc import Callable

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.domain.entities import SynthesizedResponse
from src.ports.inbound import KnowledgeQueryPort, DocumentIngestionPort
from src.ports.outbound import VectorStoragePort, GraphStoragePort, LLMInferencePort


# ─── Request Models ───────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank or whitespace-only")
        return v


# ─── Response Models ──────────────────────────────────────────────────────────


class CitationResponse(BaseModel):
    """Single citation in the query response."""

    chunk_id: str
    source_document: str
    snippet: str


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    route_used: str
    citations: list[CitationResponse]


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    status: str = "accepted"
    filename: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    neo4j: bool
    ollama: bool


class ErrorResponse(BaseModel):
    """Generic error response body."""

    detail: str = "Internal server error"


# ─── Domain-to-Response Mapping ───────────────────────────────────────────────


def _map_synthesized_response(domain_resp: SynthesizedResponse) -> QueryResponse:
    """Map domain SynthesizedResponse to HTTP QueryResponse."""
    return QueryResponse(
        answer=domain_resp.answer_text,
        route_used=domain_resp.route_used.value,
        citations=[
            CitationResponse(
                chunk_id=c.chunk_id,
                source_document=c.source_document,
                snippet=c.snippet,
            )
            for c in domain_resp.citations
        ],
    )


# ─── Adapter Class ────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


class FastAPIAdapter:
    """Inbound HTTP adapter. Receives all dependencies via constructor."""

    def __init__(
        self,
        query_service: KnowledgeQueryPort,
        ingestion_service: DocumentIngestionPort,
        vector_store: VectorStoragePort,
        graph_store: GraphStoragePort,
        llm_inference: LLMInferencePort,
        neo4j_ping: Callable[[], bool],
        ollama_ping: Callable[[], bool],
    ) -> None:
        self._query_service = query_service
        self._ingestion_service = ingestion_service
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._llm_inference = llm_inference
        self._neo4j_ping = neo4j_ping
        self._ollama_ping = ollama_ping

        self._app = FastAPI(title="FixMyPlant API")
        self._configure_cors()
        self._register_routes()

    @property
    def app(self) -> FastAPI:
        """Returns the configured FastAPI application instance."""
        return self._app

    def _configure_cors(self) -> None:
        """Register CORS middleware restricted to Streamlit origins."""
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:8501",
                "http://127.0.0.1:8501",
            ],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    def _register_routes(self) -> None:
        """Register all endpoint handlers."""

        @self._app.post("/query", response_model=QueryResponse)
        async def query_endpoint(request: QueryRequest) -> QueryResponse:
            try:
                domain_response = self._query_service.ask(request.question)
                return _map_synthesized_response(domain_response)
            except Exception as exc:
                logger.exception("Unhandled error in /query: %s", exc)
                raise HTTPException(status_code=500, detail="Internal server error")

        @self._app.post("/ingest", response_model=IngestResponse)
        async def ingest_endpoint(file: UploadFile) -> IngestResponse:
            raw_bytes = await file.read()
            filename = file.filename or "unknown"
            try:
                self._ingestion_service.ingest(raw_bytes, filename)
            except Exception as exc:
                logger.warning("Ingestion error (non-critical): %s", exc)
            return IngestResponse(status="accepted", filename=filename)

        @self._app.get("/health", response_model=HealthResponse)
        async def health_endpoint() -> HealthResponse:
            try:
                neo4j_ok = self._neo4j_ping()
            except Exception:
                neo4j_ok = False
            try:
                ollama_ok = self._ollama_ping()
            except Exception:
                ollama_ok = False
            return HealthResponse(status="ok", neo4j=neo4j_ok, ollama=ollama_ok)
