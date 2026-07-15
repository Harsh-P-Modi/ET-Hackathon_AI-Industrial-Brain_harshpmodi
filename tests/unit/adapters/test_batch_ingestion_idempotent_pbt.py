# Feature: batch-ingestion-adapter, Property 6: Idempotent Re-Ingestion
"""Property-based test verifying the BatchFileUploaderAdapter produces
identical call sequences when ingest() is called twice with the same arguments.

This confirms deterministic orchestration: given a deterministic parser and
embedding service, the adapter's output is fully reproducible.

**Validates: Requirements 6.1, 6.2**
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
    """Mock DocumentParsingPort that returns the same fixed data every call."""

    def __init__(self, chunks: list[DocumentChunk], nodes: list[EquipmentNode]) -> None:
        self._chunks = chunks
        self._nodes = nodes

    def parse(self, raw_bytes: bytes, filename: str) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
        return self._chunks, self._nodes


class RecordingLLM:
    """Mock LLMInferencePort that returns a deterministic embedding for any text."""

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}
        self._counter = 0

    def classify_route(self, query: str) -> str:
        return "vector_search"

    def synthesize(self, query: str, context: list[Any]) -> str:
        return ""

    def embed(self, text: str) -> list[float]:
        # Return the same embedding for the same text (deterministic)
        if text not in self._cache:
            self._cache[text] = [float(self._counter), 0.1, 0.2]
            self._counter += 1
        return self._cache[text]


class RecordingStorage:
    """Mock VectorStoragePort + GraphStoragePort that records (method, args) tuples."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def semantic_search(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
        return []

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        self.calls.append(("upsert_chunk", (chunk,)))

    def get_neighbors(self, equipment_id: str, depth: int) -> list[EquipmentNode]:
        return []

    def get_community_summary(self, community_id: str) -> str:
        return ""

    def upsert_equipment(self, node: EquipmentNode) -> None:
        self.calls.append(("upsert_equipment", (node,)))

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
        self.calls.append(("upsert_relationship", (from_id, to_id, rel_type)))

    def upsert_community(self, community_id: str, summary: str, member_equipment_ids: list[str]) -> None:
        self.calls.append(("upsert_community", (community_id, summary, member_equipment_ids)))


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(raw_bytes=_raw_bytes, filename=_filenames, parser_output=_parser_output)
def test_idempotent_re_ingestion(
    raw_bytes: bytes,
    filename: str,
    parser_output: tuple[list[DocumentChunk], list[EquipmentNode]],
) -> None:
    """Property 6: Calling ingest() twice with identical arguments produces
    the same sequence of storage calls with identical arguments both times.

    The parser is deterministic (returns same chunks/nodes both times) and
    the LLM embed returns the same embedding for the same text both times.
    Therefore the adapter must produce an identical call sequence.

    **Validates: Requirements 6.1, 6.2**
    """
    chunks, nodes = parser_output

    # --- First invocation ---
    parser = RecordingParser(chunks, nodes)
    llm = RecordingLLM()
    storage1 = RecordingStorage()

    adapter = BatchFileUploaderAdapter(parser=parser, llm=llm, storage=storage1)
    adapter.ingest(raw_bytes, filename)

    # --- Second invocation (same adapter, same args) ---
    storage2 = RecordingStorage()
    # Reuse same parser and llm (both deterministic) but fresh storage recorder
    adapter2 = BatchFileUploaderAdapter(parser=parser, llm=llm, storage=storage2)
    adapter2.ingest(raw_bytes, filename)

    # --- Idempotency assertions ---

    # Same number of calls
    assert len(storage1.calls) == len(storage2.calls), (
        f"Call count mismatch: first={len(storage1.calls)}, second={len(storage2.calls)}"
    )

    # Same order and same arguments
    for i, (call1, call2) in enumerate(zip(storage1.calls, storage2.calls)):
        method1, args1 = call1
        method2, args2 = call2

        assert method1 == method2, (
            f"Call #{i}: method mismatch — first='{method1}', second='{method2}'"
        )
        assert args1 == args2, (
            f"Call #{i} ({method1}): argument mismatch — "
            f"first={args1}, second={args2}"
        )
