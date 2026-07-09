"""Unit tests for _serialize_context helper method."""

from datetime import datetime

from src.domain.entities import DocumentChunk, EquipmentNode, MaintenanceEvent


def _make_adapter():
    """Create a LangGraphOrchestratorAdapter with mocked health check."""
    from unittest.mock import patch, MagicMock

    with patch("src.adapters.outbound.ollama_llm_adapter.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()

        from src.adapters.outbound.ollama_llm_adapter import (
            LangGraphOrchestratorAdapter,
        )

        adapter = LangGraphOrchestratorAdapter(
            vector_store=MagicMock(),
            graph_store=MagicMock(),
            fuser=MagicMock(),
        )
    return adapter


class TestSerializeContext:
    """Tests for _serialize_context helper."""

    def test_serialize_document_chunk(self):
        adapter = _make_adapter()
        chunk = DocumentChunk(
            chunk_id="abc123",
            parent_id="parent1",
            text="This is a test chunk about pumps.",
            embedding=None,
            source_document="maintenance_manual.pdf",
            equipment_refs=("P-101",),
        )

        result = adapter._serialize_context([chunk])

        assert "[chunk_abc123]" in result
        assert "(from: maintenance_manual.pdf)" in result
        assert "This is a test chunk about pumps." in result

    def test_serialize_equipment_node(self):
        adapter = _make_adapter()
        node = EquipmentNode(
            equipment_id="P-101",
            name="Feed Pump P-101",
            equipment_type="Pump",
            connects_to=("V-201", "HX-301"),
        )

        result = adapter._serialize_context([node])

        assert "[equip_P-101]" in result
        assert "(type: Pump)" in result
        assert "Name: Feed Pump P-101" in result
        assert "Connects to: V-201, HX-301" in result

    def test_serialize_maintenance_event(self):
        adapter = _make_adapter()
        event = MaintenanceEvent(
            event_id="evt_042",
            equipment_id="P-101",
            description="Replaced bearing assembly due to excessive vibration",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            performed_by="tech_john",
        )

        result = adapter._serialize_context([event])

        assert "[event_evt_042]" in result
        assert "(equipment: P-101" in result
        assert "date: 2024-01-15T10:30:00)" in result
        assert "Replaced bearing assembly due to excessive vibration" in result

    def test_serialize_mixed_items_separated_by_blank_line(self):
        adapter = _make_adapter()
        chunk = DocumentChunk(
            chunk_id="c1",
            parent_id="p1",
            text="Chunk text.",
            embedding=None,
            source_document="doc.pdf",
        )
        node = EquipmentNode(
            equipment_id="E1",
            name="Equip One",
            equipment_type="Valve",
            connects_to=("E2",),
        )
        event = MaintenanceEvent(
            event_id="ev1",
            equipment_id="E1",
            description="Serviced valve.",
            timestamp=datetime(2024, 6, 1),
        )

        result = adapter._serialize_context([chunk, node, event])

        # Items separated by blank lines (\n\n)
        parts = result.split("\n\n")
        assert len(parts) == 3
        assert "[chunk_c1]" in parts[0]
        assert "[equip_E1]" in parts[1]
        assert "[event_ev1]" in parts[2]

    def test_serialize_empty_list(self):
        adapter = _make_adapter()
        result = adapter._serialize_context([])
        assert result == ""

    def test_serialize_equipment_empty_connects_to(self):
        adapter = _make_adapter()
        node = EquipmentNode(
            equipment_id="X-99",
            name="Standalone Unit",
            equipment_type="Sensor",
            connects_to=(),
        )

        result = adapter._serialize_context([node])

        assert "[equip_X-99]" in result
        assert "Connects to: " in result  # empty string after join
