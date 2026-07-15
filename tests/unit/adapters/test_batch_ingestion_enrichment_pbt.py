# Feature: batch-ingestion-adapter, Property 2: Chunk Enrichment Data Preservation
"""Property-based test verifying that dataclasses.replace preserves all
original DocumentChunk fields except `embedding` when enriching a chunk.

**Validates: Requirements 2.3**
"""

import dataclasses

from hypothesis import given, settings
from hypothesis import strategies as st

from src.domain.entities import DocumentChunk


# --- Strategies ---

_text_strategy = st.text(min_size=1, max_size=200)
_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50,
)
_equipment_refs_strategy = st.tuples(
    *[st.text(min_size=1, max_size=20) for _ in range(0)]
) | st.tuples(*[st.text(min_size=1, max_size=20) for _ in range(1)]) | st.tuples(
    *[st.text(min_size=1, max_size=20) for _ in range(2)]
)

# More robust equipment_refs strategy using st.lists converted to tuple
_equipment_refs = st.lists(
    st.text(min_size=1, max_size=30), min_size=0, max_size=5
).map(tuple)

_document_chunk_strategy = st.builds(
    DocumentChunk,
    chunk_id=_id_strategy,
    parent_id=_id_strategy,
    text=_text_strategy,
    embedding=st.none(),  # Chunks start without embeddings
    source_document=_text_strategy,
    equipment_refs=_equipment_refs,
)

_embedding_strategy = st.lists(
    st.floats(allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=128,
)


# --- Property Test ---


@settings(max_examples=100)
@given(chunk=_document_chunk_strategy, new_embedding=_embedding_strategy)
def test_chunk_enrichment_preserves_all_fields_except_embedding(
    chunk: DocumentChunk, new_embedding: list[float]
) -> None:
    """Property 2: Chunk Enrichment Data Preservation.

    For any DocumentChunk with embedding=None and any generated embedding,
    applying dataclasses.replace(chunk, embedding=new_embedding) must:
    1. Preserve chunk_id, parent_id, text, source_document, equipment_refs unchanged
    2. Set the embedding field to the new_embedding value

    **Validates: Requirements 2.3**
    """
    # Act — enrich the chunk the same way the adapter does
    enriched = dataclasses.replace(chunk, embedding=new_embedding)

    # Assert — all original fields preserved
    assert enriched.chunk_id == chunk.chunk_id
    assert enriched.parent_id == chunk.parent_id
    assert enriched.text == chunk.text
    assert enriched.source_document == chunk.source_document
    assert enriched.equipment_refs == chunk.equipment_refs

    # Assert — embedding is the new value
    assert enriched.embedding == new_embedding
    assert enriched.embedding is not None
