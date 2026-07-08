"""Outbound port Protocols — define what infrastructure the domain needs.

All outbound ports are @runtime_checkable so adapters can be verified at startup
via isinstance().
"""

from typing import Protocol, runtime_checkable

from src.domain.entities import DocumentChunk, EquipmentNode, MaintenanceEvent


@runtime_checkable
class VectorStoragePort(Protocol):
    def semantic_search(self, embedding: list[float], top_k: int) -> list[DocumentChunk]: ...
    def upsert_chunk(self, chunk: DocumentChunk) -> None: ...


@runtime_checkable
class GraphStoragePort(Protocol):
    def get_neighbors(self, equipment_id: str, depth: int) -> list[EquipmentNode]: ...
    def get_community_summary(self, community_id: str) -> str: ...
    def upsert_equipment(self, node: EquipmentNode) -> None: ...
    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str) -> None: ...


@runtime_checkable
class LLMInferencePort(Protocol):
    def classify_route(self, query: str) -> str: ...
    def synthesize(self, query: str, context: list[DocumentChunk | EquipmentNode | MaintenanceEvent]) -> str: ...
    def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class DocumentParsingPort(Protocol):
    def parse(self, raw_bytes: bytes, filename: str) -> tuple[list[DocumentChunk], list[EquipmentNode]]: ...
