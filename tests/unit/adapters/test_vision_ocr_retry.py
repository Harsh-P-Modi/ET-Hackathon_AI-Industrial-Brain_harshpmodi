"""Unit tests for VisionOCRAdapter retry and error handling.

Validates:
- Transient errors (429, 5xx) trigger retry with exponential backoff (Req 8.1, 8.3)
- Permanent errors (401, 403) raise VisionOCRAdapterError immediately (Req 8.5)
- Validation failures retry then skip page (Req 4.2, 4.3)
- All retries exhausted returns None (Req 8.4)
- API errors vs validation errors are distinguished (Req 8.5)
"""

import os
from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError

from src.adapters.outbound.vision_ocr_adapter import (
    VisionOCRAdapter,
    VisionOCRAdapterError,
    MAX_RETRIES,
    BASE_DELAY_SECONDS,
    BACKOFF_MULTIPLIER,
)


@pytest.fixture
def adapter():
    """Create a VisionOCRAdapter with mocked client."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
        with patch("src.adapters.outbound.vision_ocr_adapter.genai.Client") as mock_client_cls:
            inst = VisionOCRAdapter()
            inst._client = MagicMock()
            yield inst


class TestRetryOnTransientError:
    """Transient errors should trigger retry with exponential backoff."""

    def test_retry_on_transient_error(self, adapter):
        """Mock Gemini API to raise a transient 429 error, verify retries with sleep delays."""
        transient_error = Exception("429 Resource has been exhausted (rate limit)")

        adapter._client.models.generate_content.side_effect = transient_error

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            result = adapter._call_gemini_with_retry(
                image_bytes=b"fake-image",
                mime_type="image/png",
                page=1,
                filename="test.png",
            )

        # All retries exhausted → returns None
        assert result is None
        # Should have been called MAX_RETRIES times
        assert adapter._client.models.generate_content.call_count == MAX_RETRIES
        # Sleep called for each retry except the last attempt (MAX_RETRIES - 1 sleeps)
        assert mock_sleep.call_count == MAX_RETRIES - 1
        # Verify exponential backoff delays
        for i, call in enumerate(mock_sleep.call_args_list):
            expected_delay = BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** i)
            assert call[0][0] == expected_delay

    def test_retry_succeeds_on_second_attempt(self, adapter):
        """Transient error on first call, success on second — returns valid result."""
        valid_json = '{"equipment": [], "connections": [], "text_chunks": []}'
        mock_response = MagicMock()
        mock_response.text = valid_json

        adapter._client.models.generate_content.side_effect = [
            Exception("503 Service Unavailable"),
            mock_response,
        ]

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            result = adapter._call_gemini_with_retry(
                image_bytes=b"fake-image",
                mime_type="image/png",
                page=1,
                filename="test.png",
            )

        assert result is not None
        assert result.equipment == []
        assert result.connections == []
        assert result.text_chunks == []
        # One sleep between attempts
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == BASE_DELAY_SECONDS


class TestPermanentErrorRaisesImmediately:
    """Permanent errors (401, 403) should raise VisionOCRAdapterError without retries."""

    def test_401_raises_immediately(self, adapter):
        """401 Unauthorized raises VisionOCRAdapterError on first attempt."""
        adapter._client.models.generate_content.side_effect = Exception(
            "401 Unauthorized: Invalid API key"
        )

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            with pytest.raises(VisionOCRAdapterError) as exc_info:
                adapter._call_gemini_with_retry(
                    image_bytes=b"fake-image",
                    mime_type="image/png",
                    page=1,
                    filename="test.png",
                )

        assert exc_info.value.error_type == "permanent_api_error"
        assert exc_info.value.page == 1
        # No retries — only one call
        assert adapter._client.models.generate_content.call_count == 1
        # No sleep called
        mock_sleep.assert_not_called()

    def test_403_raises_immediately(self, adapter):
        """403 Forbidden raises VisionOCRAdapterError on first attempt."""
        adapter._client.models.generate_content.side_effect = Exception(
            "403 Forbidden: Quota exhausted"
        )

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            with pytest.raises(VisionOCRAdapterError) as exc_info:
                adapter._call_gemini_with_retry(
                    image_bytes=b"fake-image",
                    mime_type="image/png",
                    page=2,
                    filename="diagram.pdf",
                )

        assert exc_info.value.error_type == "permanent_api_error"
        assert exc_info.value.page == 2
        assert adapter._client.models.generate_content.call_count == 1
        mock_sleep.assert_not_called()


class TestValidationFailureRetriesThenSkips:
    """Pydantic validation failure should retry and then return None."""

    def test_validation_failure_retries_then_skips(self, adapter):
        """Invalid JSON response triggers retry; if all fail, returns None."""
        # Return JSON with wrong types that will fail Pydantic validation
        # equipment must be a list, not a string
        mock_response = MagicMock()
        mock_response.text = '{"equipment": "not_a_list", "connections": 123, "text_chunks": false}'

        adapter._client.models.generate_content.return_value = mock_response

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            result = adapter._call_gemini_with_retry(
                image_bytes=b"fake-image",
                mime_type="image/png",
                page=1,
                filename="test.png",
            )

        # Should return None after exhausting retries
        assert result is None
        # API called MAX_RETRIES times (retries on validation failure)
        assert adapter._client.models.generate_content.call_count == MAX_RETRIES
        # Validation errors don't trigger sleep (no time.sleep for validation retries)
        mock_sleep.assert_not_called()

    def test_validation_succeeds_on_second_attempt(self, adapter):
        """First response invalid, second response valid — returns extraction."""
        invalid_response = MagicMock()
        invalid_response.text = "not json at all"

        valid_response = MagicMock()
        valid_response.text = '{"equipment": [{"name": "Pump-1", "equipment_type": "pump"}], "connections": [], "text_chunks": []}'

        adapter._client.models.generate_content.side_effect = [
            invalid_response,
            valid_response,
        ]

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            result = adapter._call_gemini_with_retry(
                image_bytes=b"fake-image",
                mime_type="image/png",
                page=1,
                filename="test.png",
            )

        assert result is not None
        assert len(result.equipment) == 1
        assert result.equipment[0].name == "Pump-1"
        # No sleep for validation retries
        mock_sleep.assert_not_called()


class TestAllRetriesExhaustedReturnsNone:
    """When all retry attempts are exhausted, _call_gemini_with_retry returns None."""

    def test_all_retries_exhausted_returns_none(self, adapter):
        """Always-failing transient errors exhaust retries → None returned."""
        adapter._client.models.generate_content.side_effect = Exception(
            "500 Internal Server Error"
        )

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            result = adapter._call_gemini_with_retry(
                image_bytes=b"fake-image",
                mime_type="image/png",
                page=3,
                filename="manual.pdf",
            )

        assert result is None
        assert adapter._client.models.generate_content.call_count == MAX_RETRIES
        assert mock_sleep.call_count == MAX_RETRIES - 1


class TestDistinguishesApiVsValidationErrors:
    """API errors use backoff+sleep; validation errors don't add sleep delays."""

    def test_api_errors_get_backoff_sleep(self, adapter):
        """Transient API errors trigger time.sleep with exponential backoff."""
        adapter._client.models.generate_content.side_effect = Exception(
            "502 Bad Gateway"
        )

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            adapter._call_gemini_with_retry(
                image_bytes=b"fake-image",
                mime_type="image/png",
                page=1,
                filename="test.png",
            )

        # API errors cause sleep between retries
        assert mock_sleep.call_count == MAX_RETRIES - 1
        # Verify backoff pattern
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        expected_delays = [
            BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** i)
            for i in range(MAX_RETRIES - 1)
        ]
        assert delays == expected_delays

    def test_validation_errors_no_sleep(self, adapter):
        """Validation errors retry without adding sleep delays."""
        mock_response = MagicMock()
        # Return text that causes Pydantic ValidationError
        mock_response.text = '{"equipment": "not_a_list"}'

        adapter._client.models.generate_content.return_value = mock_response

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            result = adapter._call_gemini_with_retry(
                image_bytes=b"fake-image",
                mime_type="image/png",
                page=1,
                filename="test.png",
            )

        # Should still retry
        assert adapter._client.models.generate_content.call_count == MAX_RETRIES
        # But no sleep delays for validation errors
        mock_sleep.assert_not_called()
        # Returns None after all validation retries
        assert result is None

    def test_mixed_api_then_validation_error(self, adapter):
        """First call API error (sleep), second call returns invalid JSON (no sleep)."""
        invalid_response = MagicMock()
        invalid_response.text = '{"equipment": "wrong_type", "connections": 999}'

        adapter._client.models.generate_content.side_effect = [
            Exception("504 Gateway Timeout"),  # API error → sleep
            invalid_response,  # Validation error → no sleep
            invalid_response,  # Validation error → no sleep (last attempt)
        ]

        with patch("src.adapters.outbound.vision_ocr_adapter.time.sleep") as mock_sleep:
            result = adapter._call_gemini_with_retry(
                image_bytes=b"fake-image",
                mime_type="image/png",
                page=1,
                filename="test.png",
            )

        assert result is None
        assert adapter._client.models.generate_content.call_count == MAX_RETRIES
        # Only one sleep — for the API error (first attempt)
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == BASE_DELAY_SECONDS
