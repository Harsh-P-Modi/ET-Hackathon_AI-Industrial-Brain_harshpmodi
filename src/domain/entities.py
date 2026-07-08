"""Domain entities for FixMyPlant knowledge intelligence system.

All entities are frozen dataclasses forming the system's shared vocabulary.
Only standard library imports are used — no vendor SDK dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RouteType(Enum):
    """Enumeration of retrieval strategies."""

    VECTOR = "vector_search"
    GRAPH_LOCAL = "graph_local_search"
    GRAPH_GLOBAL = "graph_global_search"
    HYBRID = "hybrid_fusion"


@dataclass(frozen=True)
class DocumentChunk:
    """Text segment extracted from a source document with optional embedding."""

    chunk_id: str
    parent_id: str
    text: str
    embedding: list[float] | None
    source_document: str
    equipment_refs: tuple[str, ...] = field(default_factory=tuple)

    def __hash__(self) -> int:
        return hash((
            self.chunk_id,
            self.parent_id,
            self.text,
            tuple(self.embedding) if self.embedding is not None else None,
            self.source_document,
            self.equipment_refs,
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentChunk):
            return NotImplemented
        return (
            self.chunk_id == other.chunk_id
            and self.parent_id == other.parent_id
            and self.text == other.text
            and self.embedding == other.embedding
            and self.source_document == other.source_document
            and self.equipment_refs == other.equipment_refs
        )


@dataclass(frozen=True)
class EquipmentNode:
    """Industrial equipment with typed connections to other equipment."""

    equipment_id: str
    name: str
    equipment_type: str
    connects_to: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MaintenanceEvent:
    """Recorded maintenance activity on a piece of equipment."""

    event_id: str
    equipment_id: str
    description: str
    timestamp: datetime
    performed_by: str | None = None


@dataclass(frozen=True)
class QueryPlan:
    """Classified intent of a user query."""

    raw_query: str
    route: RouteType
    target_equipment_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Citation:
    """Source reference linking an answer back to a chunk and document."""

    chunk_id: str
    source_document: str
    snippet: str


@dataclass(frozen=True)
class SynthesizedResponse:
    """Final answer with citations and route metadata."""

    answer_text: str
    citations: tuple[Citation, ...]
    route_used: RouteType
