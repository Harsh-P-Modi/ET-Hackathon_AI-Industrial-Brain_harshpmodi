"""Detect communities in the equipment graph and generate summaries.

GDS-first with connected-components fallback.

Usage:
    python scripts/rebuild_communities.py
"""
import logging
import sys
from collections import deque
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.adapters.outbound.neo4j_storage_adapter import Neo4jUnifiedStorageAdapter
from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter
from src.domain.entities import EquipmentNode
from src.domain.services import HybridContextFuser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def detect_communities_gds(driver) -> list[dict]:
    """Attempt Louvain community detection via GDS.

    Returns list of {community_id, members, context_entities}.
    Raises ClientError if GDS procedures are not available.
    """
    with driver.session() as session:
        # Project the graph
        session.run(
            "CALL gds.graph.project('equipment_graph', 'Equipment', 'CONNECTS_TO')"
        )
        # Run Louvain
        result = session.run(
            "CALL gds.louvain.stream('equipment_graph') "
            "YIELD nodeId, communityId "
            "RETURN gds.util.asNode(nodeId).equipment_id AS equipment_id, communityId"
        )
        # Group by communityId
        community_map: dict[int, list[str]] = {}
        for record in result:
            cid = record["communityId"]
            eid = record["equipment_id"]
            community_map.setdefault(cid, []).append(eid)

        # Cleanup projection
        session.run("CALL gds.graph.drop('equipment_graph')")

    communities = []
    for idx, (cid, members) in enumerate(community_map.items()):
        communities.append({
            "community_id": f"community_{idx}",
            "members": members,
            "context_entities": [
                EquipmentNode(
                    equipment_id=m, name=m, equipment_type="", connects_to=()
                )
                for m in members
            ],
        })
    return communities


def detect_communities_fallback(driver) -> list[dict]:
    """Connected-components BFS fallback when GDS is unavailable.

    Treats CONNECTS_TO edges as undirected for grouping purposes.
    """
    # Fetch all equipment nodes and edges
    with driver.session() as session:
        nodes_result = session.run(
            "MATCH (e:Equipment) "
            "RETURN e.equipment_id AS equipment_id, e.name AS name, "
            "e.equipment_type AS equipment_type"
        )
        nodes: dict[str, EquipmentNode] = {}
        for record in nodes_result:
            nodes[record["equipment_id"]] = EquipmentNode(
                equipment_id=record["equipment_id"],
                name=record["name"] or "",
                equipment_type=record["equipment_type"] or "",
                connects_to=(),
            )

        edges_result = session.run(
            "MATCH (a:Equipment)-[:CONNECTS_TO]->(b:Equipment) "
            "RETURN a.equipment_id AS from_id, b.equipment_id AS to_id"
        )
        # Build undirected adjacency list
        adjacency: dict[str, set[str]] = {eid: set() for eid in nodes}
        for record in edges_result:
            from_id = record["from_id"]
            to_id = record["to_id"]
            if from_id in adjacency:
                adjacency[from_id].add(to_id)
            if to_id in adjacency:
                adjacency[to_id].add(from_id)

    # BFS connected components
    visited: set[str] = set()
    communities: list[dict] = []
    idx = 0

    for node_id in nodes:
        if node_id in visited:
            continue
        # BFS from this node
        component: list[str] = []
        queue = deque([node_id])
        visited.add(node_id)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        communities.append({
            "community_id": f"community_{idx}",
            "members": component,
            "context_entities": [nodes[eid] for eid in component if eid in nodes],
        })
        idx += 1

    return communities


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    storage = Neo4jUnifiedStorageAdapter(driver)
    fuser = HybridContextFuser()
    llm = LangGraphOrchestratorAdapter(
        vector_store=storage, graph_store=storage, fuser=fuser
    )

    # Attempt GDS, fallback to connected-components
    try:
        communities = detect_communities_gds(driver)
        print("Community detection: GDS-based (Louvain)")
        logger.info("Community detection path: GDS-based (Louvain)")
    except (ClientError, Exception) as exc:
        logger.info(
            "GDS not available (%s), falling back to connected-components", exc
        )
        communities = detect_communities_fallback(driver)
        print("Community detection: fallback connected-components")
        logger.info("Community detection path: fallback connected-components")

    # Summarize and upsert each community
    success_count = 0
    for community in communities:
        try:
            summary = llm.synthesize(
                query=(
                    f"Summarize the role and relationships of this equipment group: "
                    f"{', '.join(community['members'])}"
                ),
                context=community["context_entities"],
            )
            storage.upsert_community(
                community_id=community["community_id"],
                summary=summary,
                member_equipment_ids=community["members"],
            )
            print(
                f"  Upserted community {community['community_id']} "
                f"({len(community['members'])} members)"
            )
            success_count += 1
        except Exception as exc:
            logger.error(
                "Failed to summarize/upsert community %s: %s",
                community["community_id"],
                exc,
            )
            print(f"  FAILED community {community['community_id']}: {exc}")

    print(f"\nRebuilt {success_count}/{len(communities)} communities.")
    driver.close()


if __name__ == "__main__":
    main()
