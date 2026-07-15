"""Unit test verifying Neo4jUnifiedStorageAdapter satisfies GraphStoragePort and VectorStoragePort.

Uses a MagicMock for the neo4j Driver — no real Neo4j needed.
Validates that the adapter class structurally satisfies both @runtime_checkable protocols,
including the new `upsert_community` method.

Validates: Requirements 3.3
"""

from unittest.mock import MagicMock

from src.adapters.outbound.neo4j_storage_adapter import Neo4jUnifiedStorageAdapter
from src.ports.outbound import GraphStoragePort, VectorStoragePort


def test_adapter_satisfies_graph_storage_port():
    """Neo4jUnifiedStorageAdapter is recognized as a GraphStoragePort instance.

    This confirms the adapter implements all required methods including
    upsert_community, per the @runtime_checkable protocol check.
    """
    mock_driver = MagicMock()
    adapter = Neo4jUnifiedStorageAdapter(mock_driver)
    assert isinstance(adapter, GraphStoragePort)


def test_adapter_satisfies_vector_storage_port():
    """Neo4jUnifiedStorageAdapter is recognized as a VectorStoragePort instance.

    This confirms the adapter implements semantic_search and upsert_chunk.
    """
    mock_driver = MagicMock()
    adapter = Neo4jUnifiedStorageAdapter(mock_driver)
    assert isinstance(adapter, VectorStoragePort)


def test_adapter_satisfies_both_ports_simultaneously():
    """A single adapter instance satisfies both ports — required for composition root wiring."""
    mock_driver = MagicMock()
    adapter = Neo4jUnifiedStorageAdapter(mock_driver)
    assert isinstance(adapter, GraphStoragePort) and isinstance(adapter, VectorStoragePort)
