"""Inbound port Protocols — define how external actors drive the system.

No @runtime_checkable on inbound ports: structural subtyping is sufficient.
Exceptions propagate unmodified.
"""

from typing import Protocol

from src.domain.entities import SynthesizedResponse


class KnowledgeQueryPort(Protocol):
    def ask(self, question: str) -> SynthesizedResponse: ...


class DocumentIngestionPort(Protocol):
    def ingest(self, raw_bytes: bytes, filename: str) -> None: ...
