"""Composition root — wires all adapters and exposes the ASGI app for uvicorn.

Run via: uvicorn src.main:app --reload
"""

import logging
import re

import requests

from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    OLLAMA_HOST,
    API_HOST,
    API_PORT,
)
from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter
from src.adapters.outbound.vision_ocr_adapter import VisionOCRAdapter
from src.adapters.outbound.composite_parser_adapter import CompositeParserAdapter
from src.adapters.outbound.inmemory_storage_adapter import InMemoryStorageAdapter
from src.adapters.inbound.batch_uploader_adapter import BatchFileUploaderAdapter
from src.adapters.inbound.fastapi_adapter import FastAPIAdapter
from src.domain.entities import Citation, RouteType, SynthesizedResponse
from src.domain.services import HybridContextFuser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        fused_context = final_state.get("fused_context") or []

        # Extract [Source: ...] citations from the answer
        from src.domain.entities import DocumentChunk

        source_citations = re.findall(r"\[Source:\s*([^\]]+)\]", answer_text)

        # Build Citation objects from all chunks in fused context that match cited sources
        chunk_list = [c for c in fused_context if isinstance(c, DocumentChunk)]

        citations: list[Citation] = []
        seen_docs: set[str] = set()

        if source_citations:
            # Map cited source labels back to chunks
            for cited_source in source_citations:
                for chunk in chunk_list:
                    source_label = chunk.source_document.replace(".txt", "").replace("_", " ").title()
                    if source_label.lower() == cited_source.strip().lower() and chunk.source_document not in seen_docs:
                        citations.append(
                            Citation(
                                chunk_id=chunk.chunk_id,
                                source_document=chunk.source_document,
                                snippet=chunk.text[:500],
                            )
                        )
                        seen_docs.add(chunk.source_document)
                        break
        elif chunk_list:
            # Fallback: if LLM didn't use [Source:] tags, cite all context chunks
            for chunk in chunk_list[:3]:
                if chunk.source_document not in seen_docs:
                    citations.append(
                        Citation(
                            chunk_id=chunk.chunk_id,
                            source_document=chunk.source_document,
                            snippet=chunk.text[:500],
                        )
                    )
                    seen_docs.add(chunk.source_document)

        # Clean up citation tags from the answer for display
        clean_answer = re.sub(r"\s*\[Source:\s*[^\]]+\]", "", answer_text).strip()

        return SynthesizedResponse(
            answer_text=clean_answer,
            citations=tuple(citations),
            route_used=route,
        )


# ─── Outbound Adapters ───────────────────────────────────────────────────────

# Try Neo4j first, fall back to in-memory storage if unavailable
try:
    from neo4j import GraphDatabase
    from src.adapters.outbound.neo4j_storage_adapter import Neo4jUnifiedStorageAdapter
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    storage_adapter = Neo4jUnifiedStorageAdapter(neo4j_driver)
    logger.info("Connected to Neo4j at %s", NEO4J_URI)
    _using_neo4j = True
except Exception as exc:
    logger.warning("Neo4j unavailable (%s) — using in-memory storage.", exc)
    neo4j_driver = None
    storage_adapter = InMemoryStorageAdapter()
    _using_neo4j = False

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
    if neo4j_driver is None:
        return False
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


# ─── Ingestion Adapter ────────────────────────────────────────────────────────

# Use composite parser: handles .txt locally, delegates images/PDFs to Gemini (if key is set)
try:
    vision_parser = VisionOCRAdapter()
except RuntimeError:
    logger.warning("GEMINI_API_KEY not set — Vision OCR disabled. Only text files can be ingested.")
    vision_parser = None

parser = CompositeParserAdapter(vision_parser=vision_parser)
ingestion_adapter = BatchFileUploaderAdapter(parser=parser, llm=llm_adapter, storage=storage_adapter)

# ─── Inbound Adapter ─────────────────────────────────────────────────────────

adapter = FastAPIAdapter(
    query_service=query_service,
    ingestion_service=ingestion_adapter,
    vector_store=storage_adapter,
    graph_store=storage_adapter,
    llm_inference=llm_adapter,
    neo4j_ping=neo4j_ping,
    ollama_ping=ollama_ping,
)

app = adapter.app


# ─── Auto-Ingest Golden Dataset on Startup (in-memory mode) ──────────────────

def _auto_ingest_golden_dataset() -> None:
    """Ingest golden dataset files at startup when using in-memory storage.

    This ensures the demo works immediately without running a separate script.
    When Neo4j is available, data is already persisted and this is skipped.
    """
    from pathlib import Path

    dataset_dir = Path(__file__).resolve().parent.parent / "data" / "golden_dataset"
    if not dataset_dir.is_dir():
        logger.info("No golden dataset found at %s — skipping auto-ingest.", dataset_dir)
        return

    files = sorted(dataset_dir.iterdir())
    logger.info("Auto-ingesting %d files from golden dataset...", len(files))

    for filepath in files:
        if filepath.is_file():
            try:
                raw_bytes = filepath.read_bytes()
                ingestion_adapter.ingest(raw_bytes, filepath.name)
                logger.info("  Ingested: %s", filepath.name)
            except Exception as exc:
                logger.warning("  Failed to ingest %s: %s", filepath.name, exc)

    if hasattr(storage_adapter, 'stats'):
        logger.info("Auto-ingest complete. Storage stats: %s", storage_adapter.stats())
    else:
        logger.info("Auto-ingest complete.")


# Run auto-ingest if using in-memory storage (data would be empty otherwise)
if not _using_neo4j:
    _auto_ingest_golden_dataset()
