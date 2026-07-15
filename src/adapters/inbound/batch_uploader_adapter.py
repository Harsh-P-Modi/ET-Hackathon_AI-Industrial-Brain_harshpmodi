"""Batch file uploader adapter — orchestrates parse -> embed -> upsert.

Implements DocumentIngestionPort. Contains no extraction or storage
logic — purely sequences calls to injected outbound ports.
"""

from dataclasses import replace
import logging

from src.domain.entities import DocumentChunk, EquipmentNode
from src.ports.outbound import (
    DocumentParsingPort,
    LLMInferencePort,
    VectorStoragePort,
    GraphStoragePort,
)

logger = logging.getLogger(__name__)


class BatchFileUploaderAdapter:
    """Orchestrates file ingestion: parse -> embed -> upsert.

    Implements DocumentIngestionPort. Contains no extraction or storage
    logic — purely sequences calls to injected outbound ports.
    """

    def __init__(
        self,
        parser: DocumentParsingPort,
        llm: LLMInferencePort,
        storage: "VectorStoragePort & GraphStoragePort",
    ) -> None:
        self._parser = parser
        self._llm = llm
        self._storage = storage

    def ingest(self, raw_bytes: bytes, filename: str) -> None:
        """Ingest a single file: parse, embed chunks, upsert all entities.

        Raises no exceptions for parse failures — logs and returns.
        Caller (CLI script) is responsible for batch-level error aggregation.
        """
        # Step 1: Parse
        try:
            chunks, equipment_nodes = self._parser.parse(raw_bytes, filename)
        except Exception as exc:
            logger.error("Failed to parse '%s': %s", filename, exc)
            return

        # Step 2: Embed each chunk and upsert
        for chunk in chunks:
            embedding = self._llm.embed(chunk.text)
            enriched_chunk = replace(chunk, embedding=embedding)
            self._storage.upsert_chunk(enriched_chunk)

        # Step 3: Upsert equipment nodes and their relationships
        for node in equipment_nodes:
            self._storage.upsert_equipment(node)
            for target_id in node.connects_to:
                self._storage.upsert_relationship(
                    node.equipment_id, target_id, "CONNECTS_TO"
                )
