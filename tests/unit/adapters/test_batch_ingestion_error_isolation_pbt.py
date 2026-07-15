# Feature: batch-ingestion-adapter, Property 7: Error Isolation in Batch Processing
"""Property-based test verifying that when processing a batch of N files where
the K-th file causes a parse exception, all other N-1 files are still processed
correctly and the exception does not propagate to the caller.

**Validates: Requirements 6.3, 6.4**
"""

from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
from hypothesis import given, settings

from src.adapters.inbound.batch_uploader_adapter import BatchFileUploaderAdapter
from src.domain.entities import DocumentChunk, EquipmentNode


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_filenames = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
).map(lambda s: s + ".pdf")

_raw_bytes = st.binary(min_size=1, max_size=64)

_document_chunks = st.builds(
    DocumentChunk,
    chunk_id=st.text(min_size=1, max_size=10),
    parent_id=st.text(min_size=1, max_size=10),
    text=st.text(min_size=1, max_size=20),
    embedding=st.none(),
    source_document=st.text(min_size=1, max_size=10),
    equipment_refs=st.just(()),
)

_equipment_nodes = st.builds(
    EquipmentNode,
    equipment_id=st.text(min_size=1, max_size=10),
    name=st.text(min_size=1, max_size=10),
    equipment_type=st.text(min_size=1, max_size=10),
    connects_to=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=2).map(
        tuple
    ),
)


# ---------------------------------------------------------------------------
# Mock ports
# ---------------------------------------------------------------------------


class FailingParser:
    """Mock DocumentParsingPort that raises on the K-th call."""

    def __init__(
        self,
        fail_index: int,
        chunks: list[DocumentChunk],
        nodes: list[EquipmentNode],
    ) -> None:
        self._fail_index = fail_index
        self._chunks = chunks
        self._nodes = nodes
        self._call_count = 0

    def parse(
        self, raw_bytes: bytes, filename: str
    ) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
        current = self._call_count
        self._call_count += 1
        if current == self._fail_index:
            raise ValueError(f"Simulated parse failure on call #{current}")
        return self._chunks, self._nodes


class StubLLM:
    """Mock LLMInferencePort returning deterministic embedding."""

    def classify_route(self, query: str) -> str:
        return "vector_search"

    def synthesize(self, query: str, context: list[Any]) -> str:
        return ""

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class CountingStorage:
    """Mock storage that counts upsert calls."""

    def __init__(self) -> None:
        self.upsert_chunk_count = 0
        self.upsert_equipment_count = 0
        self.upsert_relationship_count = 0

    def semantic_search(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
        return []

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        self.upsert_chunk_count += 1

    def get_neighbors(self, equipment_id: str, depth: int) -> list[EquipmentNode]:
        return []

    def get_community_summary(self, community_id: str) -> str:
        return ""

    def upsert_equipment(self, node: EquipmentNode) -> None:
        self.upsert_equipment_count += 1

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
        self.upsert_relationship_count += 1

    def upsert_community(
        self, community_id: str, summary: str, member_equipment_ids: list[str]
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    num_files=st.integers(min_value=2, max_value=10),
    fail_index_base=st.floats(min_value=0.0, max_value=1.0),
    chunks=st.lists(_document_chunks, min_size=1, max_size=3),
    nodes=st.lists(_equipment_nodes, min_size=0, max_size=2),
)
def test_error_isolation_in_batch_processing(
    num_files: int,
    fail_index_base: float,
    chunks: list[DocumentChunk],
    nodes: list[EquipmentNode],
) -> None:
    """Property 7: For any batch of N files where file at index K causes a
    parse exception, the adapter catches the exception internally (logs and
    returns). All other N-1 files are still processed correctly.

    **Validates: Requirements 6.3, 6.4**
    """
    # Derive fail_index from the float so it's within [0, num_files-1]
    fail_index = int(fail_index_base * (num_files - 1))
    fail_index = min(fail_index, num_files - 1)

    # Set up mocks
    parser = FailingParser(fail_index=fail_index, chunks=chunks, nodes=nodes)
    llm = StubLLM()
    storage = CountingStorage()

    adapter = BatchFileUploaderAdapter(parser=parser, llm=llm, storage=storage)

    # --- Simulate batch processing (as the CLI script does) ---
    # Call ingest() for each file in the batch — NO exception should propagate
    for i in range(num_files):
        # The adapter catches parse exceptions internally and returns normally
        adapter.ingest(b"file_content", f"file_{i}.pdf")

    # --- Assertions ---
    successful_files = num_files - 1

    # Expected storage calls for N-1 successful files
    expected_chunk_upserts = successful_files * len(chunks)
    expected_equipment_upserts = successful_files * len(nodes)
    expected_relationship_upserts = successful_files * sum(
        len(node.connects_to) for node in nodes
    )

    # Verify: all N-1 non-failing files had their chunks upserted
    assert storage.upsert_chunk_count == expected_chunk_upserts, (
        f"Expected {expected_chunk_upserts} upsert_chunk calls "
        f"({successful_files} files × {len(chunks)} chunks), "
        f"got {storage.upsert_chunk_count}. "
        f"N={num_files}, fail_index={fail_index}"
    )

    # Verify: all N-1 non-failing files had their equipment nodes upserted
    assert storage.upsert_equipment_count == expected_equipment_upserts, (
        f"Expected {expected_equipment_upserts} upsert_equipment calls "
        f"({successful_files} files × {len(nodes)} nodes), "
        f"got {storage.upsert_equipment_count}. "
        f"N={num_files}, fail_index={fail_index}"
    )

    # Verify: all N-1 non-failing files had their relationships upserted
    assert storage.upsert_relationship_count == expected_relationship_upserts, (
        f"Expected {expected_relationship_upserts} upsert_relationship calls, "
        f"got {storage.upsert_relationship_count}. "
        f"N={num_files}, fail_index={fail_index}"
    )
