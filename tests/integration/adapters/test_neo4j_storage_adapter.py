"""Integration tests for Neo4jUnifiedStorageAdapter.

Implements every item from the Testing Checklist (Spec 01 §6).
Requires a running Neo4j instance (docker-compose.local.yml).

Run with:
    pytest tests/integration/adapters/test_neo4j_storage_adapter.py -v
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

# Set test environment before importing config
os.environ["ENVIRONMENT"] = "test"
os.environ["NEO4J_PASSWORD"] = os.environ.get("NEO4J_PASSWORD", "testpassword")

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.adapters.outbound.neo4j_storage_adapter import Neo4jUnifiedStorageAdapter
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.domain.entities import DocumentChunk, EquipmentNode


def _neo4j_available() -> bool:
    """Check if Neo4j is reachable."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j is not available at {NEO4J_URI}",
)


@pytest.fixture(scope="session")
def neo4j_driver():
    """Session-scoped Neo4j driver fixture."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    yield driver
    driver.close()


@pytest.fixture
def adapter(neo4j_driver):
    """Per-test adapter instance."""
    return Neo4jUnifiedStorageAdapter(neo4j_driver)


@pytest.fixture
def test_id():
    """Unique test run ID to namespace test data."""
    return uuid.uuid4().hex[:8]


@pytest.fixture(autouse=True)
def cleanup(neo4j_driver, test_id):
    """Clean up test nodes after each test."""
    yield
    with neo4j_driver.session() as session:
        # Clean up test Equipment, Chunk, Community, MaintenanceLog nodes with test prefix
        session.run(
            "MATCH (n) WHERE n.equipment_id STARTS WITH $prefix "
            "OR n.chunk_id STARTS WITH $prefix "
            "OR n.community_id STARTS WITH $prefix "
            "OR n.event_id STARTS WITH $prefix "
            "DETACH DELETE n",
            parameters={"prefix": f"test-{test_id}"},
        )


class TestGetNeighbors:
    """§6 Checklist item 1: get_neighbors returns connected nodes."""

    def test_depth_1_returns_connected_node(self, adapter, test_id):
        """Insert 2 Equipment nodes + CONNECTS_TO edge; get_neighbors depth=1 returns the connected node."""
        pump = EquipmentNode(
            equipment_id=f"test-{test_id}-P101",
            name="Pump P-101",
            equipment_type="Pump",
            connects_to=(),
        )
        valve = EquipmentNode(
            equipment_id=f"test-{test_id}-V201",
            name="Valve V-201",
            equipment_type="Valve",
            connects_to=(),
        )

        adapter.upsert_equipment(pump)
        adapter.upsert_equipment(valve)
        adapter.upsert_relationship(pump.equipment_id, valve.equipment_id, "CONNECTS_TO")

        neighbors = adapter.get_neighbors(pump.equipment_id, depth=1)

        assert len(neighbors) == 1
        assert neighbors[0].equipment_id == valve.equipment_id
        assert neighbors[0].name == "Valve V-201"
        assert neighbors[0].equipment_type == "Valve"


class TestSemanticSearch:
    """§6 Checklist item 2: semantic_search returns chunks by descending score."""

    def test_returns_chunks_ordered_by_score(self, adapter, neo4j_driver, test_id):
        """Upsert 2 chunks with different embeddings; search with one returns it first."""
        # Create a 768-dim embedding (all zeros except one dimension)
        embedding_a = [0.0] * 768
        embedding_a[0] = 1.0  # Pointing in dimension 0

        embedding_b = [0.0] * 768
        embedding_b[1] = 1.0  # Pointing in dimension 1

        chunk_a = DocumentChunk(
            chunk_id=f"test-{test_id}-chunk-a",
            parent_id=f"test-{test_id}-doc",
            text="Safety protocols for high-pressure valves",
            embedding=embedding_a,
            source_document="manual.pdf",
            equipment_refs=(),
        )
        chunk_b = DocumentChunk(
            chunk_id=f"test-{test_id}-chunk-b",
            parent_id=f"test-{test_id}-doc",
            text="Maintenance schedule for Q4",
            embedding=embedding_b,
            source_document="schedule.pdf",
            equipment_refs=(),
        )

        adapter.upsert_chunk(chunk_a)
        adapter.upsert_chunk(chunk_b)

        # Search with embedding_a — chunk_a should rank first (exact match in cosine)
        results = adapter.semantic_search(embedding=embedding_a, top_k=2)

        assert len(results) >= 1
        assert results[0].chunk_id == chunk_a.chunk_id


class TestIdempotentIngestion:
    """§6 Checklist item 3: re-running ingestion does not create duplicates."""

    def test_upsert_equipment_twice_no_duplicates(self, adapter, neo4j_driver, test_id):
        """MERGE-based upsert creates one node even when called twice."""
        node = EquipmentNode(
            equipment_id=f"test-{test_id}-P999",
            name="Pump P-999",
            equipment_type="Pump",
            connects_to=(),
        )

        adapter.upsert_equipment(node)
        adapter.upsert_equipment(node)  # Second call — should not duplicate

        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (e:Equipment {equipment_id: $eid}) RETURN count(e) AS cnt",
                parameters={"eid": node.equipment_id},
            )
            record = result.single()
            assert record["cnt"] == 1

    def test_upsert_chunk_twice_no_duplicates(self, adapter, neo4j_driver, test_id):
        """MERGE-based upsert creates one chunk even when called twice."""
        chunk = DocumentChunk(
            chunk_id=f"test-{test_id}-chunk-dup",
            parent_id="parent-1",
            text="Test chunk for duplication check",
            embedding=[0.1] * 768,
            source_document="test.pdf",
            equipment_refs=(),
        )

        adapter.upsert_chunk(chunk)
        adapter.upsert_chunk(chunk)  # Second call

        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (c:Chunk {chunk_id: $cid}) RETURN count(c) AS cnt",
                parameters={"cid": chunk.chunk_id},
            )
            record = result.single()
            assert record["cnt"] == 1


class TestDimensionMismatch:
    """§6 Checklist item 4: dimension mismatch raises a clear error."""

    def test_wrong_embedding_dimension_raises_valueerror(self, adapter, test_id):
        """Embedding with wrong dimension count raises ValueError, not silent failure."""
        bad_chunk = DocumentChunk(
            chunk_id=f"test-{test_id}-chunk-bad",
            parent_id="parent-1",
            text="This chunk has a wrong-size embedding",
            embedding=[0.5] * 512,  # 512 instead of 768
            source_document="bad.pdf",
            equipment_refs=(),
        )

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            adapter.upsert_chunk(bad_chunk)


class TestMaintenanceLogSanityCheck:
    """plan.md Phase 2 sanity check: Equipment correctly points to MaintenanceLog."""

    def test_equipment_maintained_by_maintenance_log(self, adapter, neo4j_driver, test_id):
        """Create Equipment + MaintenanceLog with MAINTAINED_BY; verify relationship."""
        equip_id = f"test-{test_id}-P100"
        event_id = f"test-{test_id}-ML001"

        with neo4j_driver.session() as session:
            # Create Equipment node
            session.run(
                "MERGE (e:Equipment {equipment_id: $eid}) "
                "SET e.name = $name, e.equipment_type = $etype",
                parameters={"eid": equip_id, "name": "Test Pump", "etype": "Pump"},
            )
            # Create MaintenanceLog node
            session.run(
                "MERGE (m:MaintenanceLog {event_id: $event_id}) "
                "SET m.description = $desc, m.timestamp = $ts, m.performed_by = $by",
                parameters={
                    "event_id": event_id,
                    "desc": "Replaced seal",
                    "ts": "2024-01-15T10:00:00",
                    "by": "Tech-A",
                },
            )
            # Create MAINTAINED_BY relationship
            session.run(
                "MATCH (e:Equipment {equipment_id: $eid}) "
                "MATCH (m:MaintenanceLog {event_id: $event_id}) "
                "MERGE (e)-[:MAINTAINED_BY]->(m)",
                parameters={"eid": equip_id, "event_id": event_id},
            )

            # Verify relationship exists
            result = session.run(
                "MATCH (e:Equipment {equipment_id: $eid})-[:MAINTAINED_BY]->(m:MaintenanceLog) "
                "RETURN m.event_id AS event_id, m.description AS description",
                parameters={"eid": equip_id},
            )
            record = result.single()
            assert record is not None
            assert record["event_id"] == event_id
            assert record["description"] == "Replaced seal"
