# Feature: batch-ingestion-adapter, Property 1: Ingestion Sequencing Invariant
"""Property-based test verifying the BatchFileUploaderAdapter always calls
operations in strict order: parse() → embed() → upsert_chunk() for each chunk,
and equipment upserts come after parse.

**Validates: Requirements 2.1, 2.2, 2.4, 2.7**
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

import hypothesis.strategies as st
from hypothesis import given, settings

from src.adapters.inbound.batch_uploader_adapter import BatchFileUploaderAdapter
from src.domain.entities import DocumentChunk, EquipmentNode


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Random filenames: non-empty printable strings with a file extension
_filenames = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=30,
).map(lambda s: s + ".pdf")

# Random raw bytes (simulate file content)
_raw_bytes = st.binary(min_size=1, max_size=256)

# Strategy to build a DocumentChunk with no embedding (as parser would return)
_document_chunks = st.builds(
    DocumentChunk,
    chunk_id=st.text(min_size=1, max_size=20),
    parent_id=st.text(min_size=1, max_size=20),
    text=st.text(min_size=1, max_size=100),
    embedding=st.none(),
    source_document=st.text(min_size=1, max_size=30),
    equipment_refs=st.tuples(st.text(min_size=1, max_size=10)).map(tuple),
)

# Strategy to build an EquipmentNode with varying connects_to
_equipment_nodes = st.builds(
    EquipmentNode,
    equipment_id=st.text(min_size=1, max_size=20),
    name=st.text(min_size=1, max_size=30),
    equipment_type=st.text(min_size=1, max_size=20),
    connects_to=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5).map(
        tuple
    ),
)

# Combined strategy for parser output
_parser_output = st.tuples(
    st.lists(_document_chunks, min_size=0, max_size=5),
    st.lists(_equipment_nodes, min_size=0, max_size=5),
)


# ---------------------------------------------------------------------------
# Recording mock ports
# ---------------------------------------------------------------------------


class RecordingParser:
    """Mock DocumentParsingPort that records calls and returns generated data."""

    def __init__(self, chunks: list[DocumentChunk], nodes: list[EquipmentNode]) -> None:
        self._chunks = chunks
        self._nodes = nodes
        self.calls: list[str] = []

    def parse(self, raw_bytes: bytes, filename: str) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
        self.calls.append("parse")
        return self._chunks, self._nodes


class RecordingLLM:
    """Mock LLMInferencePort that records embed calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify_route(self, query: str) -> str:
        return "vector_search"

    def synthesize(self, query: str, context: list[Any]) -> str:
        return ""

    def embed(self, text: str) -> list[float]:
        self.calls.append("embed")
        return [0.1, 0.2, 0.3]


class RecordingStorage:
    """Mock VectorStoragePort + GraphStoragePort that records all upsert calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def semantic_search(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
        return []

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        self.calls.append("upsert_chunk")

    def get_neighbors(self, equipment_id: str, depth: int) -> list[EquipmentNode]:
        return []

    def get_community_summary(self, community_id: str) -> str:
        return ""

    def upsert_equipment(self, node: EquipmentNode) -> None:
        self.calls.append("upsert_equipment")

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
        self.calls.append("upsert_relationship")

    def upsert_community(self, community_id: str, summary: str, member_equipment_ids: list[str]) -> None:
        self.calls.append("upsert_community")


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(raw_bytes=_raw_bytes, filename=_filenames, parser_output=_parser_output)
def test_ingestion_sequencing_invariant(
    raw_bytes: bytes,
    filename: str,
    parser_output: tuple[list[DocumentChunk], list[EquipmentNode]],
) -> None:
    """Property 1: For any valid input where parsing succeeds, the adapter
    calls parse() first, then for each chunk embed() before upsert_chunk(),
    and all equipment operations come after parse().

    **Validates: Requirements 2.1, 2.2, 2.4, 2.7**
    """
    chunks, nodes = parser_output

    # Set up recording mocks
    parser = RecordingParser(chunks, nodes)
    llm = RecordingLLM()
    storage = RecordingStorage()

    adapter = BatchFileUploaderAdapter(parser=parser, llm=llm, storage=storage)
    adapter.ingest(raw_bytes, filename)

    # Collect the unified call log in order
    # We interleave based on the known implementation structure:
    # parse -> (embed, upsert_chunk)* -> (upsert_equipment, upsert_relationship*)*
    # But we verify ordering invariants from recorded calls.

    # Build a unified timeline from the three recorders
    # Each recorder appends in call order; we reconstruct the global order
    # by replaying from a single shared log.
    # Actually, let's use a shared call log approach instead:

    # Re-run with a shared log
    shared_log: list[str] = []

    class SharedParser:
        def parse(self, raw_bytes: bytes, filename: str) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
            shared_log.append("parse")
            return chunks, nodes

    class SharedLLM:
        def classify_route(self, query: str) -> str:
            return "vector_search"

        def synthesize(self, query: str, context: list[Any]) -> str:
            return ""

        def embed(self, text: str) -> list[float]:
            shared_log.append("embed")
            return [0.1, 0.2, 0.3]

    class SharedStorage:
        def semantic_search(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
            return []

        def upsert_chunk(self, chunk: DocumentChunk) -> None:
            shared_log.append("upsert_chunk")

        def get_neighbors(self, equipment_id: str, depth: int) -> list[EquipmentNode]:
            return []

        def get_community_summary(self, community_id: str) -> str:
            return ""

        def upsert_equipment(self, node: EquipmentNode) -> None:
            shared_log.append("upsert_equipment")

        def upsert_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
            shared_log.append("upsert_relationship")

        def upsert_community(self, community_id: str, summary: str, member_equipment_ids: list[str]) -> None:
            shared_log.append("upsert_community")

    adapter2 = BatchFileUploaderAdapter(
        parser=SharedParser(), llm=SharedLLM(), storage=SharedStorage()
    )
    adapter2.ingest(raw_bytes, filename)

    # --- Invariant assertions ---

    # 1. parse() must be the very first call (Req 2.1)
    if shared_log:
        assert shared_log[0] == "parse", (
            f"First call must be 'parse', got '{shared_log[0]}'"
        )

    # 2. Every embed() must appear before its corresponding upsert_chunk() (Req 2.2, 2.4)
    #    Since chunks are processed sequentially, embed/upsert_chunk pairs alternate.
    embed_count = 0
    upsert_chunk_count = 0
    for call in shared_log:
        if call == "embed":
            embed_count += 1
        elif call == "upsert_chunk":
            upsert_chunk_count += 1
            # At any point, we must have seen at least as many embeds as upsert_chunks
            assert embed_count >= upsert_chunk_count, (
                f"upsert_chunk called (#{upsert_chunk_count}) before corresponding "
                f"embed (only {embed_count} embeds seen so far). Log: {shared_log}"
            )

    # 3. Total embed calls equals total upsert_chunk calls equals number of chunks
    assert embed_count == len(chunks), (
        f"Expected {len(chunks)} embed calls, got {embed_count}"
    )
    assert upsert_chunk_count == len(chunks), (
        f"Expected {len(chunks)} upsert_chunk calls, got {upsert_chunk_count}"
    )

    # 4. No embed() or upsert_chunk() before parse() (Req 2.7)
    parse_index = shared_log.index("parse") if "parse" in shared_log else -1
    for i, call in enumerate(shared_log):
        if call in ("embed", "upsert_chunk", "upsert_equipment", "upsert_relationship"):
            assert i > parse_index, (
                f"Operation '{call}' at index {i} occurred before parse at index {parse_index}"
            )

    # 5. Equipment upserts happen after parse (Req 2.7)
    equipment_calls = [
        i for i, c in enumerate(shared_log) if c in ("upsert_equipment", "upsert_relationship")
    ]
    if equipment_calls:
        assert all(idx > parse_index for idx in equipment_calls), (
            "Equipment upsert operations must come after parse()"
        )

    # 6. Total upsert_equipment calls equals number of equipment nodes
    equipment_upsert_count = sum(1 for c in shared_log if c == "upsert_equipment")
    assert equipment_upsert_count == len(nodes), (
        f"Expected {len(nodes)} upsert_equipment calls, got {equipment_upsert_count}"
    )

    # 7. Total upsert_relationship calls equals sum of all connects_to entries
    expected_relationships = sum(len(n.connects_to) for n in nodes)
    relationship_count = sum(1 for c in shared_log if c == "upsert_relationship")
    assert relationship_count == expected_relationships, (
        f"Expected {expected_relationships} upsert_relationship calls, got {relationship_count}"
    )
