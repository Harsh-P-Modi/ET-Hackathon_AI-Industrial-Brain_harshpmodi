# Feature: batch-ingestion-adapter, Property 3: Equipment Persistence Completeness
"""Property-based test verifying equipment persistence completeness.

For any list of EquipmentNodes returned by the parser, the adapter SHALL call
`upsert_equipment()` exactly once per node AND call
`upsert_relationship(node.equipment_id, target_id, "CONNECTS_TO")` exactly once
for each `target_id` in each node's `connects_to` tuple.

**Validates: Requirements 2.5, 2.6**
"""

from unittest.mock import Mock, call

from hypothesis import given, settings
from hypothesis import strategies as st

from src.adapters.inbound.batch_uploader_adapter import BatchFileUploaderAdapter
from src.domain.entities import EquipmentNode


# --- Strategies ---

# Generate valid equipment IDs (non-empty printable strings)
equipment_ids = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=30,
)

# Generate EquipmentNode instances with varying connects_to tuples
equipment_nodes_strategy = st.builds(
    EquipmentNode,
    equipment_id=equipment_ids,
    name=st.text(min_size=1, max_size=50),
    equipment_type=st.text(min_size=1, max_size=30),
    connects_to=st.tuples(*[equipment_ids for _ in range(0)]).map(tuple)
    | st.tuples(equipment_ids).map(tuple)
    | st.tuples(equipment_ids, equipment_ids).map(tuple)
    | st.tuples(equipment_ids, equipment_ids, equipment_ids).map(tuple),
)

# Generate a list of EquipmentNodes (0 to 10 nodes)
equipment_node_lists = st.lists(equipment_nodes_strategy, min_size=0, max_size=10)


class TestEquipmentPersistenceCompleteness:
    """Property 3: Equipment Persistence Completeness.

    **Validates: Requirements 2.5, 2.6**

    For any list of EquipmentNodes returned by the parser, the adapter SHALL:
    - Call `upsert_equipment()` exactly once per node
    - Call `upsert_relationship(node.equipment_id, target_id, "CONNECTS_TO")`
      exactly once for each target_id in each node's connects_to tuple
    """

    @given(nodes=equipment_node_lists)
    @settings(max_examples=100)
    def test_upsert_equipment_called_once_per_node(
        self, nodes: list[EquipmentNode]
    ) -> None:
        """upsert_equipment is called exactly once per EquipmentNode."""
        # Arrange: mock parser returns empty chunks + generated equipment nodes
        mock_parser = Mock()
        mock_parser.parse.return_value = ([], nodes)

        mock_llm = Mock()
        mock_storage = Mock()

        adapter = BatchFileUploaderAdapter(
            parser=mock_parser,
            llm=mock_llm,
            storage=mock_storage,
        )

        # Act
        adapter.ingest(b"dummy content", "test_file.pdf")

        # Assert: upsert_equipment called exactly once per node
        assert mock_storage.upsert_equipment.call_count == len(nodes)

        expected_equipment_calls = [call(node) for node in nodes]
        assert mock_storage.upsert_equipment.call_args_list == expected_equipment_calls

    @given(nodes=equipment_node_lists)
    @settings(max_examples=100)
    def test_upsert_relationship_called_once_per_connects_to_entry(
        self, nodes: list[EquipmentNode]
    ) -> None:
        """upsert_relationship is called exactly once per connects_to entry."""
        # Arrange: mock parser returns empty chunks + generated equipment nodes
        mock_parser = Mock()
        mock_parser.parse.return_value = ([], nodes)

        mock_llm = Mock()
        mock_storage = Mock()

        adapter = BatchFileUploaderAdapter(
            parser=mock_parser,
            llm=mock_llm,
            storage=mock_storage,
        )

        # Act
        adapter.ingest(b"dummy content", "test_file.pdf")

        # Assert: upsert_relationship called once per target_id in each node
        expected_total_relationships = sum(
            len(node.connects_to) for node in nodes
        )
        assert (
            mock_storage.upsert_relationship.call_count
            == expected_total_relationships
        )

        expected_relationship_calls = [
            call(node.equipment_id, target_id, "CONNECTS_TO")
            for node in nodes
            for target_id in node.connects_to
        ]
        assert (
            mock_storage.upsert_relationship.call_args_list
            == expected_relationship_calls
        )

    @given(nodes=equipment_node_lists)
    @settings(max_examples=100)
    def test_embed_not_called_when_no_chunks(
        self, nodes: list[EquipmentNode]
    ) -> None:
        """llm.embed is never called when parser returns no chunks."""
        # Arrange: mock parser returns empty chunks + generated equipment nodes
        mock_parser = Mock()
        mock_parser.parse.return_value = ([], nodes)

        mock_llm = Mock()
        mock_storage = Mock()

        adapter = BatchFileUploaderAdapter(
            parser=mock_parser,
            llm=mock_llm,
            storage=mock_storage,
        )

        # Act
        adapter.ingest(b"dummy content", "test_file.pdf")

        # Assert: embed never called since no chunks
        mock_llm.embed.assert_not_called()
