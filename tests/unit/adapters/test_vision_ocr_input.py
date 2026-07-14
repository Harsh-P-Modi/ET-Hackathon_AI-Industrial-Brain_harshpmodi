"""Unit tests for VisionOCRAdapter input handling paths."""

import os
from unittest.mock import patch, MagicMock

import pytest

from src.adapters.outbound.vision_ocr_adapter import VisionOCRAdapter


@pytest.fixture
def adapter():
    """Create a VisionOCRAdapter with mocked client."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
        with patch("src.adapters.outbound.vision_ocr_adapter.genai.Client") as mock_client_cls:
            adapter = VisionOCRAdapter()
            # Replace internal client with a fresh MagicMock for test control
            adapter._client = MagicMock()
            yield adapter


def _mock_extraction_response(equipment=None, connections=None, text_chunks=None):
    """Build a mock Gemini response with valid extraction JSON."""
    import json

    data = {
        "equipment": equipment or [],
        "connections": connections or [],
        "text_chunks": text_chunks or [],
    }
    mock_response = MagicMock()
    mock_response.text = json.dumps(data)
    return mock_response


class TestImagePassthrough:
    """PNG/JPEG images should pass through without PDF rasterization."""

    def test_image_passthrough_no_conversion_png(self, adapter):
        """PNG image should not trigger _rasterize_pdf."""
        with patch(
            "src.adapters.outbound.vision_ocr_adapter._rasterize_pdf"
        ) as mock_rasterize:
            adapter._client.models.generate_content.return_value = (
                _mock_extraction_response(
                    equipment=[{"name": "Pump-101", "equipment_type": "pump"}],
                    text_chunks=[{"text": "Main pump", "referenced_equipment": []}],
                )
            )

            result = adapter.parse(b"fake-png-bytes", "diagram.png")

            mock_rasterize.assert_not_called()
            # Should still return valid results
            chunks, equipment = result
            assert len(chunks) > 0
            assert len(equipment) > 0

    def test_image_passthrough_no_conversion_jpeg(self, adapter):
        """JPEG image should not trigger _rasterize_pdf."""
        with patch(
            "src.adapters.outbound.vision_ocr_adapter._rasterize_pdf"
        ) as mock_rasterize:
            adapter._client.models.generate_content.return_value = (
                _mock_extraction_response(
                    equipment=[{"name": "Valve-A", "equipment_type": "valve"}],
                    text_chunks=[{"text": "Isolation valve", "referenced_equipment": []}],
                )
            )

            result = adapter.parse(b"fake-jpeg-bytes", "photo.jpeg")

            mock_rasterize.assert_not_called()
            chunks, equipment = result
            assert len(chunks) > 0
            assert len(equipment) > 0


class TestPdfRasterization:
    """PDF files should trigger rasterization via _rasterize_pdf."""

    def test_pdf_triggers_rasterization(self, adapter):
        """PDF input should call _rasterize_pdf to convert pages to images."""
        with patch(
            "src.adapters.outbound.vision_ocr_adapter._rasterize_pdf"
        ) as mock_rasterize:
            # Simulate a 2-page PDF rasterized into 2 PNG byte arrays
            mock_rasterize.return_value = [b"page1-png", b"page2-png"]

            adapter._client.models.generate_content.return_value = (
                _mock_extraction_response(
                    equipment=[{"name": "Tank-01", "equipment_type": "tank"}],
                    text_chunks=[{"text": "Storage tank", "referenced_equipment": []}],
                )
            )

            result = adapter.parse(b"fake-pdf-bytes", "process_flow.pdf")

            mock_rasterize.assert_called_once_with(b"fake-pdf-bytes")
            # Should process both pages (2 API calls)
            assert adapter._client.models.generate_content.call_count == 2
            chunks, equipment = result
            assert len(chunks) > 0
            assert len(equipment) > 0


class TestEmptyConnections:
    """Non-P&ID documents with empty connections list should not cause errors."""

    def test_empty_connections_not_error(self, adapter):
        """A valid extraction with no connections (e.g., maintenance log) is not an error."""
        adapter._client.models.generate_content.return_value = (
            _mock_extraction_response(
                equipment=[{"name": "Compressor-C1", "equipment_type": "compressor"}],
                connections=[],  # No connections — typical for non-P&ID docs
                text_chunks=[
                    {
                        "text": "Compressor C1 maintenance scheduled for Q2",
                        "referenced_equipment": ["Compressor-C1"],
                    }
                ],
            )
        )

        chunks, equipment = adapter.parse(b"fake-image-bytes", "maintenance_log.png")

        # Should succeed without error
        assert len(chunks) > 0
        assert len(equipment) == 1
        assert equipment[0].name == "Compressor-C1"
        # Equipment should have empty connects_to since no connections exist
        assert equipment[0].connects_to == ()


class TestCorruptedAndUnsupportedInput:
    """Unsupported or corrupted inputs should return empty tuples gracefully."""

    def test_corrupted_input_returns_empty_tuple(self, adapter):
        """If PDF rasterization fails (corrupted file), return ([], [])."""
        with patch(
            "src.adapters.outbound.vision_ocr_adapter._rasterize_pdf"
        ) as mock_rasterize:
            mock_rasterize.side_effect = Exception("Corrupted PDF: unable to read")

            chunks, equipment = adapter.parse(b"corrupted-pdf-bytes", "broken.pdf")

            assert chunks == []
            assert equipment == []

    def test_unsupported_extension_returns_empty(self, adapter):
        """File with unsupported extension (e.g., .docx) returns ([], [])."""
        chunks, equipment = adapter.parse(b"some-docx-bytes", "document.docx")

        assert chunks == []
        assert equipment == []
