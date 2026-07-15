"""Integration test: end-to-end ingest-then-query via HTTP layer.

Validates: Requirements 7.1

Verifies that a file ingested via POST /ingest is queryable afterward
via POST /query. Uses mocked outbound ports (no real Neo4j/Ollama) to
keep the test deterministic and fast.
"""

import io

from fastapi.testclient import TestClient

from src.adapters.inbound.batch_uploader_adapter import BatchFileUploaderAdapter
from src.adapters.inbound.fastapi_adapter import FastAPIAdapter
from src.domain.entities import (
    Citation,
    DocumentChunk,
    EquipmentNode,
    RouteType,
    SynthesizedResponse,
)


# ─── Mock Outbound Ports ──────────────────────────────────────────────────────


class MockParser:
    """Returns a known DocumentChunk and EquipmentNode for any input."""

    def parse(
        self, raw_bytes: bytes, filename: str
    ) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
        chunk = DocumentChunk(
            chunk_id="chunk_001",
            parent_id="parent_001",
            text="Replace the turbine bearing every 5000 hours of operation.",
            embedding=None,
            source_document=filename,
            equipment_refs=("EQ-TURB-01",),
        )
        node = EquipmentNode(
            equipment_id="EQ-TURB-01",
            name="Main Turbine",
            equipment_type="turbine",
            connects_to=("EQ-GEN-01",),
        )
        return [chunk], [node]


class MockLLM:
    """Returns a fixed embedding for embed() and a known answer for synthesize()."""

    FIXED_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]

    def classify_route(self, query: str) -> str:
        return "vector_search"

    def synthesize(self, query: str, context: list) -> str:
        if context:
            return f"Based on the documents: {context[0].text}"
        return "No relevant context found."

    def embed(self, text: str) -> list[float]:
        return self.FIXED_EMBEDDING


class MockStorage:
    """In-memory storage that stores upserted chunks and returns them on semantic_search."""

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.equipment: list[EquipmentNode] = []
        self.relationships: list[tuple[str, str, str]] = []
        self.communities: list[tuple[str, str, list[str]]] = []

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        # Replace if same chunk_id exists, else append
        self.chunks = [c for c in self.chunks if c.chunk_id != chunk.chunk_id]
        self.chunks.append(chunk)

    def semantic_search(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
        # Return all stored chunks (up to top_k)
        return self.chunks[:top_k]

    def upsert_equipment(self, node: EquipmentNode) -> None:
        self.equipment = [e for e in self.equipment if e.equipment_id != node.equipment_id]
        self.equipment.append(node)

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
        self.relationships.append((from_id, to_id, rel_type))

    def get_neighbors(self, equipment_id: str, depth: int) -> list[EquipmentNode]:
        return [e for e in self.equipment if e.equipment_id == equipment_id]

    def get_community_summary(self, community_id: str) -> str:
        for cid, summary, _ in self.communities:
            if cid == community_id:
                return summary
        return ""

    def upsert_community(
        self, community_id: str, summary: str, member_equipment_ids: list[str]
    ) -> None:
        self.communities.append((community_id, summary, member_equipment_ids))


class MockQueryService:
    """Query service backed by the same mock storage, producing answers from stored chunks."""

    def __init__(self, storage: MockStorage, llm: MockLLM) -> None:
        self._storage = storage
        self._llm = llm

    def ask(self, question: str) -> SynthesizedResponse:
        # Embed the question and search
        embedding = self._llm.embed(question)
        results = self._storage.semantic_search(embedding, top_k=3)

        if results:
            context_text = results[0].text
            answer = f"Based on ingested data: {context_text}"
            citations = tuple(
                Citation(
                    chunk_id=chunk.chunk_id,
                    source_document=chunk.source_document,
                    snippet=chunk.text[:100],
                )
                for chunk in results
            )
        else:
            answer = "No relevant information found."
            citations = ()

        return SynthesizedResponse(
            answer_text=answer,
            citations=citations,
            route_used=RouteType.VECTOR,
        )


# ─── Test ─────────────────────────────────────────────────────────────────────


def test_ingest_then_query_end_to_end():
    """A file ingested via /ingest is queryable afterward via /query.

    Validates: Requirements 7.1
    """
    # Arrange: wire up all mock dependencies
    mock_parser = MockParser()
    mock_llm = MockLLM()
    mock_storage = MockStorage()

    # Ingestion service: BatchFileUploaderAdapter with mocked ports
    ingestion_service = BatchFileUploaderAdapter(
        parser=mock_parser,
        llm=mock_llm,
        storage=mock_storage,
    )

    # Query service: uses the same mock storage so ingested data is visible
    query_service = MockQueryService(storage=mock_storage, llm=mock_llm)

    # Build FastAPIAdapter with all dependencies
    adapter = FastAPIAdapter(
        query_service=query_service,
        ingestion_service=ingestion_service,
        vector_store=mock_storage,
        graph_store=mock_storage,
        llm_inference=mock_llm,
        neo4j_ping=lambda: True,
        ollama_ping=lambda: True,
    )

    client = TestClient(adapter.app)

    # Act: Ingest a file via POST /ingest
    file_content = b"Turbine maintenance manual content - replace bearing every 5000 hours"
    response_ingest = client.post(
        "/ingest",
        files={"file": ("turbine_manual.pdf", io.BytesIO(file_content), "application/pdf")},
    )

    # Assert: ingest accepted
    assert response_ingest.status_code == 200
    ingest_data = response_ingest.json()
    assert ingest_data["status"] == "accepted"
    assert ingest_data["filename"] == "turbine_manual.pdf"

    # Verify the chunk was actually stored (sanity check on mock wiring)
    assert len(mock_storage.chunks) == 1
    assert mock_storage.chunks[0].chunk_id == "chunk_001"
    assert mock_storage.chunks[0].embedding == MockLLM.FIXED_EMBEDDING

    # Act: Query for content from the ingested file via POST /query
    response_query = client.post(
        "/query",
        json={"question": "How often should I replace the turbine bearing?"},
    )

    # Assert: query returns content from the ingested file
    assert response_query.status_code == 200
    query_data = response_query.json()
    assert "answer" in query_data
    assert "turbine bearing" in query_data["answer"].lower() or "5000 hours" in query_data["answer"]
    assert query_data["route_used"] == "vector_search"
    assert len(query_data["citations"]) >= 1
    assert query_data["citations"][0]["source_document"] == "turbine_manual.pdf"
    assert query_data["citations"][0]["chunk_id"] == "chunk_001"


def test_query_before_ingest_returns_empty():
    """Querying before any ingestion returns no relevant information.

    This ensures the test isn't passing due to hardcoded responses.
    """
    # Arrange: fresh mocks with no ingested data
    mock_parser = MockParser()
    mock_llm = MockLLM()
    mock_storage = MockStorage()

    ingestion_service = BatchFileUploaderAdapter(
        parser=mock_parser,
        llm=mock_llm,
        storage=mock_storage,
    )

    query_service = MockQueryService(storage=mock_storage, llm=mock_llm)

    adapter = FastAPIAdapter(
        query_service=query_service,
        ingestion_service=ingestion_service,
        vector_store=mock_storage,
        graph_store=mock_storage,
        llm_inference=mock_llm,
        neo4j_ping=lambda: True,
        ollama_ping=lambda: True,
    )

    client = TestClient(adapter.app)

    # Act: Query without any prior ingestion
    response_query = client.post(
        "/query",
        json={"question": "How often should I replace the turbine bearing?"},
    )

    # Assert: returns empty/no-info response
    assert response_query.status_code == 200
    query_data = response_query.json()
    assert query_data["citations"] == []
    assert "no relevant" in query_data["answer"].lower()
