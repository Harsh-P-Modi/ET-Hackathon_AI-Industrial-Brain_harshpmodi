"""TextDocumentParserAdapter — implements DocumentParsingPort for plain text files.

Splits text documents into overlapping chunks and extracts equipment IDs via regex.
Used for maintenance logs, specifications, and other text-based industrial documents.
"""

from __future__ import annotations

import logging
import re

from src.domain.entities import DocumentChunk, EquipmentNode

logger = logging.getLogger(__name__)

# Regex to find equipment IDs like CW-P-101, STG-101, BFW-T-201, K-501, etc.
EQUIPMENT_ID_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\b")

# Chunk splitting parameters
CHUNK_SIZE_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200


def _slugify_id(name: str, index: int) -> str:
    """Generate a deterministic ID from name and index."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return f"{slug}_c{index}"


def _split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks of roughly CHUNK_SIZE_CHARS."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start + CHUNK_SIZE_CHARS // 2:
                end = newline_pos + 2
            else:
                # Look for sentence break
                sentence_pos = text.rfind(". ", start, end)
                if sentence_pos > start + CHUNK_SIZE_CHARS // 2:
                    end = sentence_pos + 2

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        start = end - CHUNK_OVERLAP_CHARS
        if start < 0:
            start = 0
        # Prevent infinite loop if chunk_size is too small
        if end >= len(text):
            break

    return chunks


def _extract_equipment_ids(text: str) -> list[str]:
    """Extract unique equipment IDs from text."""
    matches = EQUIPMENT_ID_PATTERN.findall(text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            unique.append(match)
    return unique


def _infer_equipment_type(equipment_id: str) -> str:
    """Infer equipment type from common naming conventions."""
    type_map = {
        "P": "pump",
        "E": "heat_exchanger",
        "T": "tank",
        "V": "valve",
        "K": "compressor",
        "B": "boiler",
        "FCV": "flow_control_valve",
        "TCV": "temperature_control_valve",
        "FT": "flow_transmitter",
        "TT": "temperature_transmitter",
        "PT": "pressure_transmitter",
        "LSH": "level_switch_high",
        "LSL": "level_switch_low",
        "GD": "gas_detector",
        "FD": "flame_detector",
        "HD": "heat_detector",
        "TD": "toxic_detector",
        "DV": "deluge_valve",
        "STG": "steam_turbine",
        "GEN": "generator",
        "CEP": "condensate_pump",
        "LO": "lube_oil_system",
        "GOV": "governor",
        "VE": "vacuum_ejector",
        "TP": "turning_gear",
        "CP": "condensate_polishing",
        "DA": "deaerator",
        "BFW": "boiler_feedwater",
        "CW": "cooling_water",
        "FP": "fire_pump",
        "RV": "recycle_valve",
        "SV": "suction_valve",
    }
    # Try multi-character prefix first (e.g., BFW, CW, STG)
    parts = equipment_id.split("-")
    if len(parts) >= 2:
        prefix = parts[0]
        # Try 2-part prefix: "CW-P" → lookup "P"
        tag = parts[1] if len(parts) > 2 else parts[0]
        if prefix in type_map:
            return type_map[prefix]
        if tag in type_map:
            return type_map[tag]
    return "equipment"


def _extract_connections(text: str, equipment_ids: list[str]) -> list[tuple[str, str]]:
    """Extract connections from arrow notation (→) in text."""
    connections: list[tuple[str, str]] = []
    # Find lines with → arrows
    for line in text.split("\n"):
        if "→" in line:
            # Extract all equipment IDs in this line in order
            ids_in_line = EQUIPMENT_ID_PATTERN.findall(line)
            for i in range(len(ids_in_line) - 1):
                if ids_in_line[i] in equipment_ids and ids_in_line[i + 1] in equipment_ids:
                    connections.append((ids_in_line[i], ids_in_line[i + 1]))
    return connections


class TextDocumentParserAdapter:
    """Implements DocumentParsingPort for plain text (.txt) files.

    Splits documents into overlapping chunks, extracts equipment IDs,
    infers equipment types, and builds connection topology from arrow notation.
    """

    def parse(self, raw_bytes: bytes, filename: str) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
        """Parse a text document into structured domain entities."""
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw_bytes.decode("latin-1")
            except Exception as e:
                logger.warning("Cannot decode file %s: %s", filename, e)
                return ([], [])

        if not text.strip():
            return ([], [])

        filename_stem = filename.rsplit(".", 1)[0] if "." in filename else filename

        # Extract all equipment IDs from the full document
        all_equipment_ids = _extract_equipment_ids(text)

        # Extract connections from arrow notation
        connections = _extract_connections(text, all_equipment_ids)

        # Build connection map: equipment_id → list of target IDs
        connection_map: dict[str, list[str]] = {eid: [] for eid in all_equipment_ids}
        for from_id, to_id in connections:
            if from_id in connection_map:
                connection_map[from_id].append(to_id)

        # Split text into chunks
        chunk_texts = _split_into_chunks(text)

        # Build parent chunk (full document summary — first 500 chars)
        parent_chunk_id = _slugify_id(filename_stem, 0)
        parent_chunk = DocumentChunk(
            chunk_id=parent_chunk_id,
            parent_id="",
            text=text[:500],
            embedding=None,
            source_document=filename,
            equipment_refs=tuple(_slugify_id(eid, 0) for eid in all_equipment_ids[:10]),
        )

        # Build child chunks
        child_chunks: list[DocumentChunk] = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{parent_chunk_id}_c{i}"
            chunk_equip_ids = _extract_equipment_ids(chunk_text)
            equipment_refs = tuple(_slugify_id(eid, 0) for eid in chunk_equip_ids)
            child_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_chunk_id,
                    text=chunk_text,
                    embedding=None,
                    source_document=filename,
                    equipment_refs=equipment_refs,
                )
            )

        # Build equipment nodes
        equipment_nodes: list[EquipmentNode] = []
        for eid in all_equipment_ids:
            connects_to = tuple(
                _slugify_id(target, 0) for target in connection_map.get(eid, [])
            )
            equipment_nodes.append(
                EquipmentNode(
                    equipment_id=_slugify_id(eid, 0),
                    name=eid,
                    equipment_type=_infer_equipment_type(eid),
                    connects_to=connects_to,
                )
            )

        all_chunks = [parent_chunk] + child_chunks
        logger.info(
            "Parsed %s: %d chunks, %d equipment nodes, %d connections",
            filename, len(all_chunks), len(equipment_nodes), len(connections),
        )
        return (all_chunks, equipment_nodes)
