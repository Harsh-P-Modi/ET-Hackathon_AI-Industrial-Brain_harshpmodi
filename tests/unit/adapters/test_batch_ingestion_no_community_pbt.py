# Feature: batch-ingestion-adapter, Property 5: Ingest Never Triggers Community Operations
"""Property-based test verifying that ingest() never triggers community operations.

For any call to ingest(raw_bytes, filename) with any input, the adapter SHALL NOT
invoke upsert_community() or get_community_summary() — community rebuild is always
a manual, explicit step and never triggered by ingest().

**Validates: Requirements 5.2**
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from src.adapters.inbound.batch_uploader_adapter import BatchFileUploaderAdapter
from src.domain.entities import DocumentChunk, EquipmentNode


# --- Strategies ---

random_bytes = st.binary(min_size=0, max_size=1024)

_FILENAME_SAFE_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_."
)
random_filenames = st.text(
    alphabet=_FILENAME_SAFE_CHARS,
    min_size=1,
    max_size=80,
)

# Strategies for generating DocumentChunks and EquipmentNodes returned by the parser mock
_chunk_strategy = st.builds(
    DocumentChunk,
    chunk_id=st.text(min_size=1, max_size=20),
    parent_id=st.text(min_size=1, max_size=20),
    text=st.text(min_size=1, max_size=100),
    embedding=st.none(),
    source_document=st.text(min_size=1, max_size=50),
    equipment_refs=st.tuples(),
)

_equipment_strategy = st.builds(
    EquipmentNode,
    equipment_id=st.text(min_size=1, max_size=20),
    name=st.text(min_size=1, max_size=30),
    equipment_type=st.text(min_size=1, max_size=20),
    connects_to=st.tuples(
        *[]  # empty default
    ),
)

# Generate variable-length connects_to tuples
_equipment_with_connections = st.builds(
    EquipmentNode,
    equipment_id=st.text(min_size=1, max_size=20),
    name=st.text(min_size=1, max_size=30),
    equipment_type=st.text(min_size=1, max_size=20),
    connects_to=st.lists(
        st.text(min_size=1, max_size=20), min_size=0, max_size=5
    ).map(tuple),
)

_parsed_chunks = st.lists(_chunk_strategy, min_size=0, max_size=5)
_parsed_equipment = st.lists(_equipment_with_connections, min_size=0, max_size=5)


def _make_adapter(
    parser_return: tuple[list[DocumentChunk], list[EquipmentNode]],
) -> tuple[BatchFileUploaderAdapter, MagicMock]:
    """Build a BatchFileUploaderAdapter with all ports mocked.

    Returns the adapter and the storage mock for assertions.
    """
    parser = MagicMock()
    parser.parse.return_value = parser_return

    llm = MagicMock()
    llm.embed.return_value = [0.1, 0.2, 0.3]

    storage = MagicMock()
    # Ensure community methods are explicitly tracked as MagicMock
    storage.upsert_community = MagicMock()
    storage.get_community_summary = MagicMock()

    adapter = BatchFileUploaderAdapter(parser=parser, llm=llm, storage=storage)
    return adapter, storage


class TestIngestNeverTriggersCommunityOperations:
    """Property 5: Ingest Never Triggers Community Operations.

    **Validates: Requirements 5.2**

    For any call to ingest(raw_bytes, filename) with any input, the adapter
    SHALL NOT invoke upsert_community() or get_community_summary().
    Community rebuild is always a manual, explicit step.
    """

    @given(
        raw=random_bytes,
        filename=random_filenames,
        chunks=_parsed_chunks,
        equipment=_parsed_equipment,
    )
    @settings(max_examples=100)
    def test_upsert_community_never_called(
        self,
        raw: bytes,
        filename: str,
        chunks: list[DocumentChunk],
        equipment: list[EquipmentNode],
    ) -> None:
        """upsert_community is never called during ingest(), regardless of input."""
        adapter, storage = _make_adapter((chunks, equipment))

        adapter.ingest(raw, filename)

        storage.upsert_community.assert_not_called()

    @given(
        raw=random_bytes,
        filename=random_filenames,
        chunks=_parsed_chunks,
        equipment=_parsed_equipment,
    )
    @settings(max_examples=100)
    def test_get_community_summary_never_called(
        self,
        raw: bytes,
        filename: str,
        chunks: list[DocumentChunk],
        equipment: list[EquipmentNode],
    ) -> None:
        """get_community_summary is never called during ingest(), regardless of input."""
        adapter, storage = _make_adapter((chunks, equipment))

        adapter.ingest(raw, filename)

        storage.get_community_summary.assert_not_called()
