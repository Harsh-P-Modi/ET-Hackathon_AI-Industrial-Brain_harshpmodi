"""CompositeParserAdapter — delegates parsing to the correct adapter based on file type.

Implements DocumentParsingPort. Routes .txt files to TextDocumentParserAdapter and
image/PDF files to VisionOCRAdapter.
"""

from __future__ import annotations

import logging

from src.domain.entities import DocumentChunk, EquipmentNode
from src.adapters.outbound.text_parser_adapter import TextDocumentParserAdapter
from src.adapters.outbound.vision_ocr_adapter import VisionOCRAdapter

logger = logging.getLogger(__name__)

# File extensions handled by the text parser
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log"}

# File extensions handled by the vision/OCR parser
VISION_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


class CompositeParserAdapter:
    """Routes document parsing to the appropriate adapter based on file extension.

    - .txt, .md, .csv, .log → TextDocumentParserAdapter (local, no API calls)
    - .pdf, .png, .jpg, etc. → VisionOCRAdapter (requires Gemini API key)
    """

    def __init__(self, vision_parser: VisionOCRAdapter | None = None) -> None:
        """Initialize with optional VisionOCRAdapter.

        If vision_parser is None, image/PDF files will return empty results
        with a warning (graceful degradation when GEMINI_API_KEY is not set).
        """
        self._text_parser = TextDocumentParserAdapter()
        self._vision_parser = vision_parser

    def parse(self, raw_bytes: bytes, filename: str) -> tuple[list[DocumentChunk], list[EquipmentNode]]:
        """Parse a document by delegating to the appropriate parser."""
        ext = self._get_extension(filename)

        if ext in TEXT_EXTENSIONS:
            return self._text_parser.parse(raw_bytes, filename)
        elif ext in VISION_EXTENSIONS:
            if self._vision_parser is None:
                logger.warning(
                    "Vision parser not available (missing GEMINI_API_KEY?). "
                    "Skipping file: %s", filename
                )
                return ([], [])
            return self._vision_parser.parse(raw_bytes, filename)
        else:
            logger.warning("Unsupported file extension '%s' for file: %s", ext, filename)
            return ([], [])

    @staticmethod
    def _get_extension(filename: str) -> str:
        """Extract lowercase file extension including the dot."""
        if "." in filename:
            return "." + filename.rsplit(".", 1)[-1].lower()
        return ""
