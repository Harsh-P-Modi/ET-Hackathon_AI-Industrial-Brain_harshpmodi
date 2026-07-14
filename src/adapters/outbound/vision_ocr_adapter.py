"""VisionOCRAdapter — implements DocumentParsingPort using Google Gemini multimodal extraction.

Transforms raw industrial documents (P&ID diagrams, maintenance logs, safety manuals) into
structured DocumentChunk and EquipmentNode domain entities via single-pass multimodal extraction.
"""

from __future__ import annotations

import logging
import os
import re
import time
from enum import Enum
from io import BytesIO

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from pdf2image import convert_from_bytes

from src.domain.entities import DocumentChunk, EquipmentNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Pydantic schemas (adapter-private — never exposed to domain layer)
# ---------------------------------------------------------------------------


class _FlowDirection(str, Enum):
    """Direction of flow between two connected equipment items."""

    A_TO_B = "A_to_B"
    B_TO_A = "B_to_A"
    UNKNOWN = "unknown"


class _EquipmentExtraction(BaseModel):
    """Single equipment item extracted from a page."""

    name: str = Field(description="Equipment label as shown on the diagram")
    equipment_type: str = Field(description="Type: pump, valve, tank, sensor, etc.")


class _ConnectionExtraction(BaseModel):
    """Directional connection between two equipment items."""

    from_name: str = Field(description="Source equipment name")
    to_name: str = Field(description="Destination equipment name")
    flow_direction: _FlowDirection = Field(
        default=_FlowDirection.UNKNOWN,
        description="Flow direction: A_to_B, B_to_A, or unknown",
    )


class _TextChunkExtraction(BaseModel):
    """Text segment extracted from the document page."""

    text: str = Field(description="Extracted text content")
    referenced_equipment: list[str] = Field(
        default_factory=list,
        description="Equipment names this text refers to",
    )


class _ExtractionSchema(BaseModel):
    """Top-level extraction schema for a single page."""

    equipment: list[_EquipmentExtraction] = Field(default_factory=list)
    connections: list[_ConnectionExtraction] = Field(default_factory=list)
    text_chunks: list[_TextChunkExtraction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class VisionOCRAdapterError(Exception):
    """Raised for permanent/unrecoverable VisionOCR adapter errors."""

    def __init__(self, message: str, *, error_type: str = "unknown", page: int | None = None) -> None:
        self.error_type = error_type
        self.page = page
        super().__init__(message)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _slugify_id(name: str, page: int) -> str:
    """Generate a deterministic equipment/chunk ID.

    - Lowercase the name
    - Replace non-alphanumeric characters with hyphens
    - Collapse multiple hyphens
    - Strip leading/trailing hyphens
    - Append page number suffix
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return f"{slug}_p{page}"


def _rasterize_pdf(raw_bytes: bytes) -> list[bytes]:
    """Convert PDF bytes to a list of PNG byte arrays — one per page."""
    images = convert_from_bytes(raw_bytes)
    png_pages: list[bytes] = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        png_pages.append(buffer.getvalue())
    return png_pages


_MIME_TYPE_MAP: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def _detect_mime_type(filename: str) -> str:
    """Determine MIME type from filename extension.

    Raises ValueError for unsupported extensions.
    """
    ext = "." + filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    mime = _MIME_TYPE_MAP.get(ext)
    if mime is None:
        raise ValueError(
            f"Unsupported file extension '{ext}' for file '{filename}'. "
            f"Supported: {', '.join(sorted(_MIME_TYPE_MAP.keys()))}"
        )
    return mime


def _build_extraction_prompt() -> str:
    """Return the system instruction string for the Gemini extraction task."""
    return (
        "You are an industrial document analysis system. "
        "Analyze the provided image of an industrial document (P&ID diagram, maintenance log, or safety manual). "
        "Extract ALL of the following information:\n\n"
        "1. **Equipment**: Every piece of equipment visible (pumps, valves, tanks, sensors, "
        "heat exchangers, compressors, etc.). Provide the label/name exactly as shown and classify its type.\n\n"
        "2. **Connections**: All connections/piping between equipment. For each connection, identify "
        "the source equipment (from_name), destination equipment (to_name), and flow direction "
        "(A_to_B if flow goes from source to destination, B_to_A if reversed, unknown if not determinable).\n\n"
        "3. **Text Chunks**: All readable text segments on the page (notes, annotations, labels, "
        "specifications, warnings). For each text segment, list any equipment names it references.\n\n"
        "Be thorough — extract every piece of equipment and connection visible. "
        "Use exact labels as shown on the document. "
        "Return results in the specified JSON schema."
    )


# ---------------------------------------------------------------------------
# Domain entity mapping
# ---------------------------------------------------------------------------


def _map_to_domain_entities(
    extraction: _ExtractionSchema, filename: str, page: int
) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
    """Transform a validated extraction schema into domain entities.

    Creates a parent chunk (full page text) and child chunks (individual text segments),
    plus EquipmentNode instances with their connection topology.

    Args:
        extraction: Validated Pydantic extraction result for one page.
        filename: Original source document filename.
        page: Page number (used for deterministic ID generation).

    Returns:
        Tuple of (all_chunks, all_equipment_nodes) where all_chunks includes
        the parent chunk followed by child chunks.
    """
    # Derive the parent chunk ID from the filename stem (no extension)
    filename_stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    parent_chunk_id = _slugify_id(filename_stem, page)

    # --- Parent chunk: concatenation of all text on the page ---
    parent_text = "\n".join(tc.text for tc in extraction.text_chunks)
    parent_chunk = DocumentChunk(
        chunk_id=parent_chunk_id,
        parent_id="",
        text=parent_text,
        embedding=None,
        source_document=filename,
        equipment_refs=tuple(),
    )

    # --- Child chunks: one per text extraction ---
    child_chunks: list[DocumentChunk] = []
    for index, text_chunk in enumerate(extraction.text_chunks):
        child_id = f"{parent_chunk_id}_c{index}"
        equipment_refs = tuple(
            _slugify_id(name, page) for name in text_chunk.referenced_equipment
        )
        child_chunks.append(
            DocumentChunk(
                chunk_id=child_id,
                parent_id=parent_chunk_id,
                text=text_chunk.text,
                embedding=None,
                source_document=filename,
                equipment_refs=equipment_refs,
            )
        )

    # --- Equipment nodes with connects_to topology ---
    equipment_nodes: list[EquipmentNode] = []
    for equip in extraction.equipment:
        connects_to = tuple(
            _slugify_id(conn.to_name, page)
            for conn in extraction.connections
            if conn.from_name == equip.name
        )
        equipment_nodes.append(
            EquipmentNode(
                equipment_id=_slugify_id(equip.name, page),
                name=equip.name,
                equipment_type=equip.equipment_type,
                connects_to=connects_to,
            )
        )

    all_chunks = [parent_chunk] + child_chunks
    return (all_chunks, equipment_nodes)


# ---------------------------------------------------------------------------
# Retry / backoff constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class VisionOCRAdapter:
    """Implements DocumentParsingPort using Google Gemini multimodal extraction."""

    def __init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required. "
                "Set it in your .env file or export it before running the application."
            )
        self._model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=api_key)
        logger.info("VisionOCRAdapter initialized with model=%s", self._model)

    def _call_gemini_with_retry(
        self, image_bytes: bytes, mime_type: str, page: int, filename: str
    ) -> _ExtractionSchema | None:
        """Call Gemini API with exponential backoff retry logic.

        Returns validated _ExtractionSchema or None if page should be skipped.
        Raises VisionOCRAdapterError for permanent errors (401, 403).
        """
        prompt_text = _build_extraction_prompt()

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[
                        prompt_text,
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=_ExtractionSchema,
                    ),
                )

                # Validate response with Pydantic
                try:
                    result = _ExtractionSchema.model_validate_json(response.text)
                    return result
                except ValidationError as ve:
                    # Validation failure: retry on next loop iteration
                    logger.warning(
                        "Pydantic validation failed — filename=%s page=%d attempt=%d detail=%s",
                        filename, page, attempt + 1, str(ve)[:200],
                    )
                    if attempt < MAX_RETRIES - 1:
                        continue
                    return None

            except Exception as e:
                error_msg = str(e)
                # Check for permanent errors
                if any(code in error_msg for code in ("401", "403")):
                    raise VisionOCRAdapterError(
                        f"Permanent API error for file '{filename}' page {page}: {error_msg[:200]}",
                        error_type="permanent_api_error",
                        page=page,
                    )

                # Transient error — retry with backoff
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
                    logger.warning(
                        "Transient error, retrying — filename=%s page=%d attempt=%d delay=%.1fs error=%s",
                        filename, page, attempt + 1, delay, error_msg[:200],
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "All retries exhausted — filename=%s page=%d error=%s",
                        filename, page, error_msg[:200],
                    )
                    return None

        return None

    def parse(self, raw_bytes: bytes, filename: str) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
        """Parse a document into structured domain entities.

        Handles PDF (multi-page rasterization) and image (passthrough) inputs.
        Processes each page independently with fault isolation.
        Returns aggregated results across all pages.
        """
        # Detect input type
        try:
            mime_type = _detect_mime_type(filename)
        except ValueError as e:
            logger.warning("Unsupported file type — filename=%s error=%s", filename, str(e))
            return ([], [])

        # Prepare page images
        try:
            if mime_type == "application/pdf":
                page_images = _rasterize_pdf(raw_bytes)
                image_mime = "image/png"  # rasterized to PNG
            else:
                page_images = [raw_bytes]  # single image passthrough
                image_mime = mime_type
        except Exception as e:
            logger.warning(
                "Failed to prepare input — filename=%s error_type=input_error detail=%s",
                filename, str(e)[:200],
            )
            return ([], [])

        # Process each page independently
        all_chunks: list[DocumentChunk] = []
        all_equipment: list[EquipmentNode] = []

        for page_num, image_bytes in enumerate(page_images, start=1):
            try:
                extraction = self._call_gemini_with_retry(
                    image_bytes=image_bytes,
                    mime_type=image_mime,
                    page=page_num,
                    filename=filename,
                )

                if extraction is None:
                    logger.warning(
                        "Skipping page — filename=%s page=%d reason=extraction_failed",
                        filename, page_num,
                    )
                    continue

                chunks, equipment = _map_to_domain_entities(extraction, filename, page_num)
                all_chunks.extend(chunks)
                all_equipment.extend(equipment)

            except VisionOCRAdapterError:
                raise  # Permanent errors propagate
            except Exception as e:
                logger.warning(
                    "Page processing failed — filename=%s page=%d error_type=unexpected detail=%s",
                    filename, page_num, str(e)[:200],
                )
                continue

        return (all_chunks, all_equipment)
