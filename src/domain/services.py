"""Domain services for FixMyPlant knowledge intelligence system.

Contains QueryRouter (keyword prefilter + LLM fallback classification)
and HybridContextFuser (Weighted Reciprocal Rank Fusion).

Only standard library imports — no vendor SDK dependencies.
"""

import re
from typing import ClassVar, Union

from src.domain.entities import (
    DocumentChunk,
    EquipmentNode,
    MaintenanceEvent,
    QueryPlan,
    RouteType,
)
from src.ports.outbound import LLMInferencePort


def _get_domain_id(item: Union[DocumentChunk, EquipmentNode, MaintenanceEvent]) -> str:
    """Extract the canonical domain identifier for deduplication.

    - DocumentChunk → chunk_id
    - EquipmentNode → equipment_id
    - MaintenanceEvent → event_id
    - Otherwise raise TypeError
    """
    if isinstance(item, DocumentChunk):
        return item.chunk_id
    elif isinstance(item, EquipmentNode):
        return item.equipment_id
    elif isinstance(item, MaintenanceEvent):
        return item.event_id
    raise TypeError(f"Unknown item type: {type(item)}")


class QueryRouter:
    """Classifies natural-language queries into retrieval routes.

    Algorithm: keyword prefilter → LLM fallback → HYBRID default.
    """

    GRAPH_LOCAL_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "downstream",
        "upstream",
        "connected to",
        "fails",
        "affected",
        "depends on",
    )

    GRAPH_GLOBAL_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "common causes",
        "across all",
        "most frequent",
        "patterns across",
        "system-wide",
    )

    EQUIPMENT_ID_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"\b[A-Za-z]+\-[0-9]+\b"
    )

    def classify(self, query: str, llm_infer: LLMInferencePort) -> QueryPlan:
        """Classify query into a retrieval route.

        Algorithm:
        1. Extract equipment IDs via regex
        2. Check for GRAPH_LOCAL keywords (case-insensitive)
        3. Check for GRAPH_GLOBAL keywords (case-insensitive)
        4. If GRAPH_LOCAL keywords found → GRAPH_LOCAL (priority over GRAPH_GLOBAL)
        5. If GRAPH_GLOBAL keywords found → GRAPH_GLOBAL
        6. Otherwise invoke llm_infer.classify_route(query)
           - Valid RouteType string (case-insensitive) → use it
           - Invalid/exception → HYBRID fallback
        7. Return QueryPlan(raw_query=query, route=route, target_equipment_ids=ids)
        """
        query_lower = query.lower()
        equipment_ids = tuple(self.EQUIPMENT_ID_PATTERN.findall(query))

        has_graph_local = any(kw in query_lower for kw in self.GRAPH_LOCAL_KEYWORDS)
        has_graph_global = any(kw in query_lower for kw in self.GRAPH_GLOBAL_KEYWORDS)

        if has_graph_local:
            route = RouteType.GRAPH_LOCAL
        elif has_graph_global:
            route = RouteType.GRAPH_GLOBAL
        else:
            try:
                llm_result = llm_infer.classify_route(query)
                route = RouteType(llm_result.lower().strip())
            except (ValueError, Exception):
                route = RouteType.HYBRID

        return QueryPlan(
            raw_query=query,
            route=route,
            target_equipment_ids=equipment_ids,
        )


class HybridContextFuser:
    """Merges vector and graph results via Weighted Reciprocal Rank Fusion (WRRF)."""

    def fuse(
        self,
        vector_results: list[DocumentChunk],
        graph_results: list[Union[EquipmentNode, MaintenanceEvent]],
        k: int = 60,
    ) -> list[Union[DocumentChunk, EquipmentNode, MaintenanceEvent]]:
        """Merge vector and graph results via WRRF scoring.

        Algorithm:
        1. Build score map: for each item in vector_results at position i (0-based),
           score[id] += 1/(k + i+1)  [1-based rank]
        2. For each item in graph_results at position j,
           score[id] += 1/(k + j+1)
        3. Deduplicate by domain ID (chunk_id / equipment_id / event_id)
        4. Sort by descending score
        5. Tie-break: stable sort preserving original order,
           vector_results items before graph_results items
        6. Return full list (no truncation)
        """
        scores: dict[str, float] = {}
        items: dict[str, Union[DocumentChunk, EquipmentNode, MaintenanceEvent]] = {}
        source_order: dict[str, tuple[int, int]] = {}  # (list_index, position)

        for rank_0, chunk in enumerate(vector_results):
            did = _get_domain_id(chunk)
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank_0 + 1)
            if did not in items:
                items[did] = chunk
                source_order[did] = (0, rank_0)  # list 0 = vector

        for rank_0, node in enumerate(graph_results):
            did = _get_domain_id(node)
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank_0 + 1)
            if did not in items:
                items[did] = node
                source_order[did] = (1, rank_0)  # list 1 = graph

        # Sort by descending score, tie-break by source list (vector first), then position
        sorted_ids = sorted(
            items.keys(),
            key=lambda did: (-scores[did], source_order[did][0], source_order[did][1]),
        )

        return [items[did] for did in sorted_ids]
