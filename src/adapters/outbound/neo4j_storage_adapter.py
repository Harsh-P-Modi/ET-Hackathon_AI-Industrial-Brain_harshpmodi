"""Neo4j Unified Storage Adapter — implements VectorStoragePort and GraphStoragePort.

Uses the official neo4j Python driver. All Cypher queries use parameterized
queries ($param) — zero f-string interpolation of values into Cypher.
"""

import logging
from typing import Any

from neo4j import Driver

from src.config import VECTOR_DIMENSIONS
from src.domain.entities import DocumentChunk, EquipmentNode
from src.ports.outbound import GraphStoragePort, VectorStoragePort

logger = logging.getLogger(__name__)


class Neo4jUnifiedStorageAdapter:
    """Single adapter implementing both VectorStoragePort and GraphStoragePort.

    Requires a shared neo4j.Driver instance injected via constructor.
    Do not create a new driver per method call.
    """

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    # -----------------------------------------------------------------------
    # VectorStoragePort
    # -----------------------------------------------------------------------

    def semantic_search(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
        """Search for chunks by cosine similarity via the chunk_embeddings vector index.

        Uses db.index.vector.queryNodes. Maps raw Neo4j Records back to
        DocumentChunk entities — never leaks a raw Record past this method.
        """
        query = (
            "CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding) "
            "YIELD node, score "
            "RETURN node.chunk_id AS chunk_id, node.text AS text, "
            "node.parent_id AS parent_id, node.embedding AS embedding, "
            "node.source_document AS source_document, "
            "node.equipment_refs AS equipment_refs, score "
            "ORDER BY score DESC"
        )
        results: list[DocumentChunk] = []
        with self._driver.session() as session:
            records = session.run(query, parameters={"top_k": top_k, "embedding": embedding})
            for record in records:
                equipment_refs = record["equipment_refs"]
                if equipment_refs is None:
                    equipment_refs = ()
                else:
                    equipment_refs = tuple(equipment_refs)
                chunk = DocumentChunk(
                    chunk_id=record["chunk_id"],
                    parent_id=record["parent_id"] or "",
                    text=record["text"] or "",
                    embedding=record["embedding"],
                    source_document=record["source_document"] or "",
                    equipment_refs=equipment_refs,
                )
                results.append(chunk)
        return results

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        """Upsert a document chunk via MERGE on chunk_id. Idempotent.

        Startup assertion: embedding dimension must match the vector index's
        configured dimensions (768 for nomic-embed-text). Raises ValueError
        on mismatch — never fails silently.
        """
        if chunk.embedding is not None:
            if len(chunk.embedding) != VECTOR_DIMENSIONS:
                raise ValueError(
                    f"Embedding dimension mismatch: got {len(chunk.embedding)}, "
                    f"expected {VECTOR_DIMENSIONS} (configured for nomic-embed-text). "
                    f"Chunk ID: {chunk.chunk_id}"
                )

        query = (
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.text = $text, c.parent_id = $parent_id, "
            "c.embedding = $embedding, c.source_document = $source_document, "
            "c.equipment_refs = $equipment_refs"
        )
        params: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "parent_id": chunk.parent_id,
            "embedding": chunk.embedding,
            "source_document": chunk.source_document,
            "equipment_refs": list(chunk.equipment_refs),
        }
        with self._driver.session() as session:
            session.run(query, parameters=params)
        logger.debug("Upserted chunk %s", chunk.chunk_id)

    # -----------------------------------------------------------------------
    # GraphStoragePort
    # -----------------------------------------------------------------------

    def get_neighbors(self, equipment_id: str, depth: int) -> list[EquipmentNode]:
        """Traverse CONNECTS_TO edges from the given equipment up to `depth` hops.

        This is the Route B (local search) query — e.g. "if Pump P-101 fails,
        what's affected downstream".

        ASSUMPTION: CONNECTS_TO edges are written in strict upstream→downstream
        direction at ingestion time. This behavior is owned by Spec 03/06
        (extraction pipeline), which are not built yet. This assumption MUST be
        verified once Spec 03/06 land. If direction is inconsistent, downstream
        queries will silently return incomplete results.
        """
        # Neo4j does not support parameterized variable-length relationship bounds,
        # so we validate depth is a positive integer and embed it safely.
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth must be a positive integer, got: {depth}")

        # NOTE: Neo4j Cypher does not allow $param in relationship length ranges.
        # We validate depth above and use string formatting ONLY for the integer
        # range bound — this is safe because depth is validated as int, not user text.
        query = (
            f"MATCH (start:Equipment {{equipment_id: $equipment_id}})"
            f"-[:CONNECTS_TO*1..{depth}]->(downstream:Equipment) "
            f"RETURN DISTINCT downstream.equipment_id AS equipment_id, "
            f"downstream.name AS name, downstream.equipment_type AS equipment_type, "
            f"downstream.connects_to AS connects_to"
        )
        results: list[EquipmentNode] = []
        with self._driver.session() as session:
            records = session.run(query, parameters={"equipment_id": equipment_id})
            for record in records:
                connects_to = record["connects_to"]
                if connects_to is None:
                    connects_to = ()
                else:
                    connects_to = tuple(connects_to)
                node = EquipmentNode(
                    equipment_id=record["equipment_id"],
                    name=record["name"] or "",
                    equipment_type=record["equipment_type"] or "",
                    connects_to=connects_to,
                )
                results.append(node)
        return results

    def get_community_summary(self, community_id: str) -> str:
        """Retrieve a pre-computed community summary (Route C — global search).

        Reads from Community nodes. Does NOT traverse the live graph or
        summarize at query time — summaries are generated offline during
        ingestion (Spec 06).
        """
        query = (
            "MATCH (c:Community {community_id: $community_id}) "
            "RETURN c.summary AS summary"
        )
        with self._driver.session() as session:
            result = session.run(query, parameters={"community_id": community_id})
            record = result.single()
            if record is None:
                raise LookupError(
                    f"No community found with community_id: {community_id}"
                )
            return record["summary"] or ""

    def upsert_equipment(self, node: EquipmentNode) -> None:
        """Upsert an equipment node via MERGE on equipment_id. Idempotent."""
        query = (
            "MERGE (e:Equipment {equipment_id: $equipment_id}) "
            "SET e.name = $name, e.equipment_type = $equipment_type, "
            "e.connects_to = $connects_to"
        )
        params: dict[str, Any] = {
            "equipment_id": node.equipment_id,
            "name": node.name,
            "equipment_type": node.equipment_type,
            "connects_to": list(node.connects_to),
        }
        with self._driver.session() as session:
            session.run(query, parameters=params)
        logger.debug("Upserted equipment %s", node.equipment_id)

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
        """Create a relationship between two equipment nodes via MERGE. Idempotent.

        Uses MERGE on both endpoints and the relationship to ensure
        re-running ingestion does not create duplicates.
        """
        # Validate rel_type is a simple identifier (letters, underscores, digits)
        # to prevent Cypher injection through relationship type names.
        # Neo4j does not support parameterized relationship types, so we must
        # validate the input strictly.
        if not rel_type.replace("_", "").isalnum():
            raise ValueError(
                f"rel_type must be alphanumeric with underscores only, got: {rel_type!r}"
            )

        query = (
            f"MATCH (a:Equipment {{equipment_id: $from_id}}) "
            f"MATCH (b:Equipment {{equipment_id: $to_id}}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )
        with self._driver.session() as session:
            session.run(query, parameters={"from_id": from_id, "to_id": to_id})
        logger.debug("Upserted relationship %s -[%s]-> %s", from_id, rel_type, to_id)

    def upsert_community(self, community_id: str, summary: str, member_equipment_ids: list[str]) -> None:
        """Upsert a Community node and BELONGS_TO relationships. Idempotent via MERGE."""
        community_query = (
            "MERGE (c:Community {community_id: $community_id}) "
            "SET c.summary = $summary"
        )
        relationship_query = (
            "MATCH (e:Equipment {equipment_id: $equipment_id}) "
            "MATCH (c:Community {community_id: $community_id}) "
            "MERGE (e)-[:BELONGS_TO]->(c)"
        )
        with self._driver.session() as session:
            session.run(community_query, parameters={
                "community_id": community_id,
                "summary": summary,
            })
            for eq_id in member_equipment_ids:
                session.run(relationship_query, parameters={
                    "equipment_id": eq_id,
                    "community_id": community_id,
                })
        logger.debug("Upserted community %s with %d members", community_id, len(member_equipment_ids))
