"""In-Memory Storage Adapter — implements VectorStoragePort and GraphStoragePort.

Drop-in replacement for Neo4jUnifiedStorageAdapter when Neo4j is unavailable.
Stores all data in dictionaries. Performs brute-force cosine similarity for
vector search. Suitable for demos and development without Docker/Neo4j.
"""

import logging
import math
from collections import deque
from typing import Any

from src.domain.entities import DocumentChunk, EquipmentNode
from src.ports.outbound import GraphStoragePort, VectorStoragePort

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryStorageAdapter:
    """In-memory implementation of VectorStoragePort and GraphStoragePort.

    All data lives in Python dicts — lost on process exit. Use for demos only.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._equipment: dict[str, EquipmentNode] = {}
        self._relationships: dict[str, list[tuple[str, str]]] = {}  # from_id -> [(to_id, rel_type)]
        self._communities: dict[str, dict[str, Any]] = {}  # community_id -> {summary, members}

    # -----------------------------------------------------------------------
    # VectorStoragePort
    # -----------------------------------------------------------------------

    def semantic_search(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
        """Brute-force cosine similarity search over stored chunks."""
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self._chunks.values():
            if chunk.embedding is not None:
                score = _cosine_similarity(embedding, chunk.embedding)
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        """Store or update a chunk in memory."""
        self._chunks[chunk.chunk_id] = chunk
        logger.debug("Upserted chunk %s (total: %d)", chunk.chunk_id, len(self._chunks))

    # -----------------------------------------------------------------------
    # GraphStoragePort
    # -----------------------------------------------------------------------

    def get_neighbors(self, equipment_id: str, depth: int) -> list[EquipmentNode]:
        """BFS traversal over stored relationships up to `depth` hops."""
        if equipment_id not in self._equipment:
            # Try to find by matching the slugified ID
            found = False
            for eid in self._equipment:
                if eid == equipment_id:
                    found = True
                    break
            if not found:
                return []

        visited: set[str] = set()
        results: list[EquipmentNode] = []
        queue: deque[tuple[str, int]] = deque([(equipment_id, 0)])
        visited.add(equipment_id)

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            # Get outgoing edges
            edges = self._relationships.get(current_id, [])
            for target_id, _rel_type in edges:
                if target_id not in visited:
                    visited.add(target_id)
                    if target_id in self._equipment:
                        results.append(self._equipment[target_id])
                    queue.append((target_id, current_depth + 1))

        return results

    def get_community_summary(self, community_id: str) -> str:
        """Retrieve a stored community summary."""
        community = self._communities.get(community_id)
        if community is None:
            raise LookupError(f"No community found with community_id: {community_id}")
        return community.get("summary", "")

    def upsert_equipment(self, node: EquipmentNode) -> None:
        """Store or update an equipment node."""
        self._equipment[node.equipment_id] = node
        logger.debug("Upserted equipment %s (total: %d)", node.equipment_id, len(self._equipment))

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
        """Store a relationship between two equipment nodes."""
        if from_id not in self._relationships:
            self._relationships[from_id] = []
        # Avoid duplicates
        edge = (to_id, rel_type)
        if edge not in self._relationships[from_id]:
            self._relationships[from_id].append(edge)
        logger.debug("Upserted relationship %s -[%s]-> %s", from_id, rel_type, to_id)

    def upsert_community(self, community_id: str, summary: str, member_equipment_ids: list[str]) -> None:
        """Store a community with its summary and members."""
        self._communities[community_id] = {
            "summary": summary,
            "members": member_equipment_ids,
        }
        logger.debug("Upserted community %s with %d members", community_id, len(member_equipment_ids))

    # -----------------------------------------------------------------------
    # Stats (for debugging)
    # -----------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return counts of stored entities."""
        return {
            "chunks": len(self._chunks),
            "equipment": len(self._equipment),
            "relationships": sum(len(v) for v in self._relationships.values()),
            "communities": len(self._communities),
        }
