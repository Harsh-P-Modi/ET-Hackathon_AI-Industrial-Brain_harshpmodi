# Feature: batch-ingestion-adapter, Property 4: Community Detection Produces Valid Partitions
"""Property-based test verifying that the community detection fallback algorithm
(BFS connected-components) produces valid graph partitions for any random graph.

For any equipment graph with N nodes and E edges:
- Every equipment node belongs to exactly one community
- Each community's member set is non-empty
- Output conforms to the expected shape (community_id: str, members: list[str], context_entities: list)
- Union of all community members equals the set of all nodes

**Validates: Requirements 4.2, 4.3**
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock

import hypothesis.strategies as st
from hypothesis import given, settings

# Ensure ENVIRONMENT=test so config module doesn't require secrets
os.environ.setdefault("ENVIRONMENT", "test")

from scripts.rebuild_communities import detect_communities_fallback


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def random_graphs(draw: st.DrawFn) -> tuple[list[str], list[tuple[str, str]]]:
    """Generate a random graph with 1-50 nodes and random edges between them.

    Returns:
        Tuple of (list of node IDs, list of (from_id, to_id) edge tuples)
    """
    num_nodes = draw(st.integers(min_value=1, max_value=50))
    node_ids = [f"equip_{i}" for i in range(num_nodes)]

    # Generate random directed edges as pairs of node indices
    edges = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=num_nodes - 1),
                st.integers(min_value=0, max_value=num_nodes - 1),
            ),
            min_size=0,
            max_size=min(num_nodes * 3, 100),  # cap edges for performance
        )
    )

    # Convert index pairs to node ID pairs, filter self-loops
    edge_pairs = [
        (node_ids[a], node_ids[b])
        for a, b in edges
        if a != b
    ]

    return node_ids, edge_pairs


# ---------------------------------------------------------------------------
# Mock driver factory
# ---------------------------------------------------------------------------


class _MockRecord:
    """Lightweight mock for a Neo4j record supporting __getitem__."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


class _MockResult:
    """Lightweight mock for a Neo4j result that is iterable."""

    def __init__(self, records: list[_MockRecord]) -> None:
        self._records = records

    def __iter__(self):
        return iter(self._records)


def _make_mock_driver(
    node_ids: list[str], edge_pairs: list[tuple[str, str]]
) -> MagicMock:
    """Create a mock Neo4j driver that returns the given nodes and edges.

    The mock simulates:
    - session.run("MATCH (e:Equipment)...") -> records with equipment_id, name, equipment_type
    - session.run("MATCH (a:Equipment)-[:CONNECTS_TO]->...") -> records with from_id, to_id
    """
    # Build mock records for nodes
    node_records = [
        _MockRecord({
            "equipment_id": nid,
            "name": nid,
            "equipment_type": "pump",
        })
        for nid in node_ids
    ]

    # Build mock records for edges
    edge_records = [
        _MockRecord({"from_id": from_id, "to_id": to_id})
        for from_id, to_id in edge_pairs
    ]

    # Mock session.run() to return appropriate results based on the query
    def mock_run(query: str, **kwargs: Any) -> _MockResult:
        if "RETURN e.equipment_id" in query:
            return _MockResult(node_records)
        elif "RETURN a.equipment_id AS from_id" in query:
            return _MockResult(edge_records)
        return _MockResult([])

    # Build mock driver -> session context manager
    session = MagicMock()
    session.run = mock_run

    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    return driver


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(graph=random_graphs())
def test_community_detection_produces_valid_partitions(
    graph: tuple[list[str], list[tuple[str, str]]],
) -> None:
    """Property 4: For any equipment graph, the fallback connected-components
    algorithm produces communities where every node is in exactly one community,
    all communities are non-empty, and the output shape is correct.

    **Validates: Requirements 4.2, 4.3**
    """
    node_ids, edge_pairs = graph
    driver = _make_mock_driver(node_ids, edge_pairs)

    communities = detect_communities_fallback(driver)

    # --- Invariant assertions ---

    # 1. Output is a list of dicts
    assert isinstance(communities, list), "communities must be a list"

    # 2. At least one community must exist (since we have at least 1 node)
    assert len(communities) > 0, "Must produce at least one community"

    # 3. Each community has the correct shape and valid data
    for community in communities:
        assert isinstance(community, dict), "Each community must be a dict"
        assert "community_id" in community, "community must have 'community_id' key"
        assert "members" in community, "community must have 'members' key"
        assert "context_entities" in community, "community must have 'context_entities' key"

        # community_id is a string matching "community_{N}" format
        cid = community["community_id"]
        assert isinstance(cid, str), f"community_id must be str, got {type(cid)}"
        assert cid.startswith("community_"), (
            f"community_id must start with 'community_', got '{cid}'"
        )

        # members is a non-empty list of strings
        members = community["members"]
        assert isinstance(members, list), f"members must be a list, got {type(members)}"
        assert len(members) > 0, "Each community must have at least one member"
        for m in members:
            assert isinstance(m, str), f"Each member must be a str, got {type(m)}"

        # context_entities is a list corresponding to members
        ctx = community["context_entities"]
        assert isinstance(ctx, list), f"context_entities must be a list, got {type(ctx)}"
        assert len(ctx) == len(members), (
            f"context_entities length ({len(ctx)}) must match members length ({len(members)})"
        )

    # 4. Every node appears in exactly one community (valid partition)
    all_members: list[str] = []
    for community in communities:
        all_members.extend(community["members"])

    # Union of all community members equals the set of all nodes
    assert set(all_members) == set(node_ids), (
        f"Union of community members {set(all_members)} does not match "
        f"node set {set(node_ids)}"
    )

    # No duplicate assignments (each node in exactly one community)
    assert len(all_members) == len(set(all_members)), (
        f"Some nodes appear in multiple communities: "
        f"{len(all_members)} total vs {len(set(all_members))} unique"
    )

    # 5. community_id values are unique
    community_ids = [c["community_id"] for c in communities]
    assert len(community_ids) == len(set(community_ids)), (
        f"Duplicate community_ids found: {community_ids}"
    )
