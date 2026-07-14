"""Composition root — wires all adapters and exposes the ASGI app for uvicorn.

Run via: uvicorn src.main:app --reload
"""

import logging
import re

import requests
from neo4j import GraphDatabase

from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    OLLAMA_HOST,
    API_HOST,
    API_PORT,
)
from src.adapters.outbound.neo4j_storage_adapter import Neo4jUnifiedStorageAdapter
from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter
from src.adapters.inbound.fastapi_adapter import FastAPIAdapter
from src.domain.entities import Citation, RouteType, SynthesizedResponse
from src.domain.services import HybridContextFuser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Stub until Spec 06 (BatchFileUploaderAdapter) lands ─────────────────────


class StubDocumentIngestionPort:
    """Placeholder ingestion port — accepts calls without doing real work."""

    def ingest(self, raw_bytes: bytes, filename: str) -> None:
        logger.info("StubIngestion: received '%s' (%d bytes) — no-op", filename, len(raw_bytes))


# ─── Query Service Wrapper ───────────────────────────────────────────────────


class OrchestratorQueryService:
    """Wraps LangGraphOrchestratorAdapter to satisfy KnowledgeQueryPort.

    The orchestrator exposes classify_route, synthesize, and embed but does not
    directly implement ask(). This wrapper runs the full LangGraph pipeline and
    constructs a SynthesizedResponse from the final state.
    """

    def __init__(self, orchestrator: LangGraphOrchestratorAdapter) -> None:
        self._orchestrator = orchestrator

    def ask(self, question: str) -> SynthesizedResponse:
        """Run the orchestration pipeline and return a SynthesizedResponse."""
        compiled_graph = self._orchestrator._build_graph()
        initial_state = {
            "query": question,
            "route": "",
            "vector_results": [],
            "graph_results": [],
            "fused_context": [],
            "answer": "",
        }
        final_state = compiled_graph.invoke(initial_state)

        # Map route string back to RouteType enum
        route_str = final_state.get("route", "hybrid_fusion")
        try:
            route = RouteType(route_str)
        except ValueError:
            route = RouteType.HYBRID

        # Extract inline citation tags from the answer text
        answer_text = final_state.get("answer", "")
        citation_ids = re.findall(r"\[(chunk_[^\]]+)\]", answer_text)
        fused_context = final_state.get("fused_context") or []

        # Build Citation objects from referenced chunks in the fused context
        from src.domain.entities import DocumentChunk

        chunk_map = {
            c.chunk_id: c for c in fused_context if isinstance(c, DocumentChunk)
        }
        citations: list[Citation] = []
        seen_ids: set[str] = set()
        for cid in citation_ids:
            if cid not in seen_ids and cid in chunk_map:
                chunk = chunk_map[cid]
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        source_document=chunk.source_document,
                        snippet=chunk.text[:200],
                    )
                )
                seen_ids.add(cid)

        return SynthesizedResponse(
            answer_text=answer_text,
            citations=tuple(citations),
            route_used=route,
        )


# ─── Outbound Adapters ───────────────────────────────────────────────────────

neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
storage_adapter = Neo4jUnifiedStorageAdapter(neo4j_driver)
fuser = HybridContextFuser()
llm_adapter = LangGraphOrchestratorAdapter(
    vector_store=storage_adapter,
    graph_store=storage_adapter,
    fuser=fuser,
)

# ─── Query Service ───────────────────────────────────────────────────────────

query_service = OrchestratorQueryService(llm_adapter)

# ─── Health Check Callables ──────────────────────────────────────────────────


def neo4j_ping() -> bool:
    """Ping Neo4j with a trivial RETURN 1 query."""
    try:
        with neo4j_driver.session() as session:
            session.run("RETURN 1").single()
        return True
    except Exception:
        return False


def ollama_ping() -> bool:
    """Ping Ollama via GET /api/tags."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        return True
    except Exception:
        return False


# ─── Inbound Adapter ─────────────────────────────────────────────────────────

adapter = FastAPIAdapter(
    query_service=query_service,
    ingestion_service=StubDocumentIngestionPort(),
    vector_store=storage_adapter,
    graph_store=storage_adapter,
    llm_inference=llm_adapter,
    neo4j_ping=neo4j_ping,
    ollama_ping=ollama_ping,
)

app = adapter.app
