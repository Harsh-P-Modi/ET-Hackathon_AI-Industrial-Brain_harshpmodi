"""Unit tests for domain services: QueryRouter and HybridContextFuser.

Validates:
- QueryRouter.classify returns GRAPH_LOCAL for graph-local keyword queries.
- HybridContextFuser.fuse ranks items appearing in both lists above items in only one.
"""

from src.domain.entities import (
    DocumentChunk,
    EquipmentNode,
    MaintenanceEvent,
    QueryPlan,
    RouteType,
)
from src.domain.services import HybridContextFuser, QueryRouter
from src.ports.outbound import LLMInferencePort


class FakeLLMInferencePort:
    """Stub LLMInferencePort that satisfies the Protocol without real inference."""

    def classify_route(self, query: str) -> str:
        return "hybrid_fusion"

    def synthesize(
        self, query: str, context: list[DocumentChunk | EquipmentNode | MaintenanceEvent]
    ) -> str:
        return ""

    def embed(self, text: str) -> list[float]:
        return [0.0] * 128


class TestQueryRouter:
    """Tests for QueryRouter.classify."""

    def test_classify_returns_graph_local_for_failure_query(self) -> None:
        """QueryRouter.classify returns GRAPH_LOCAL for 'if pump P-101 fails, which valves are affected'."""
        router = QueryRouter()
        fake_llm = FakeLLMInferencePort()

        plan = router.classify(
            "if pump P-101 fails, which valves are affected", fake_llm
        )

        assert plan.route == RouteType.GRAPH_LOCAL
        assert plan.raw_query == "if pump P-101 fails, which valves are affected"
        assert "P-101" in plan.target_equipment_ids


class TestHybridContextFuser:
    """Tests for HybridContextFuser.fuse."""

    def test_fuse_ranks_shared_item_above_single_list_item(self) -> None:
        """An item appearing in both vector and graph results ranks higher than one in only one list."""
        fuser = HybridContextFuser()

        # Shared item: equipment_id "EQ-SHARED" appears in both lists
        shared_chunk = DocumentChunk(
            chunk_id="EQ-SHARED",
            parent_id="doc-1",
            text="Shared equipment document chunk",
            embedding=None,
            source_document="manual.pdf",
            equipment_refs=("EQ-SHARED",),
        )
        shared_node = EquipmentNode(
            equipment_id="EQ-SHARED",
            name="Shared Pump",
            equipment_type="pump",
            connects_to=(),
        )

        # Vector-only item
        vector_only_chunk = DocumentChunk(
            chunk_id="VEC-ONLY",
            parent_id="doc-2",
            text="Vector only chunk",
            embedding=None,
            source_document="report.pdf",
            equipment_refs=(),
        )

        # Graph-only item
        graph_only_node = EquipmentNode(
            equipment_id="GRAPH-ONLY",
            name="Graph Only Valve",
            equipment_type="valve",
            connects_to=(),
        )

        vector_results = [shared_chunk, vector_only_chunk]
        graph_results = [shared_node, graph_only_node]

        fused = fuser.fuse(vector_results, graph_results)

        # Find positions of items in the fused list
        fused_ids = [
            item.chunk_id if isinstance(item, DocumentChunk) else item.equipment_id
            for item in fused
        ]

        shared_pos = fused_ids.index("EQ-SHARED")
        vec_only_pos = fused_ids.index("VEC-ONLY")
        graph_only_pos = fused_ids.index("GRAPH-ONLY")

        # Shared item should rank above both single-list items
        assert shared_pos < vec_only_pos, (
            f"Shared item at position {shared_pos} should rank above "
            f"vector-only item at position {vec_only_pos}"
        )
        assert shared_pos < graph_only_pos, (
            f"Shared item at position {shared_pos} should rank above "
            f"graph-only item at position {graph_only_pos}"
        )
