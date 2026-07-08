# src.ports - Port Protocol definitions (inbound and outbound)

from src.ports.inbound import DocumentIngestionPort, KnowledgeQueryPort
from src.ports.outbound import (
    DocumentParsingPort,
    GraphStoragePort,
    LLMInferencePort,
    VectorStoragePort,
)

__all__ = [
    # Inbound ports
    "KnowledgeQueryPort",
    "DocumentIngestionPort",
    # Outbound ports
    "VectorStoragePort",
    "GraphStoragePort",
    "LLMInferencePort",
    "DocumentParsingPort",
]
