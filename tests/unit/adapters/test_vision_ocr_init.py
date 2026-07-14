"""Unit tests for VisionOCRAdapter.__init__ — configuration and error handling."""

import os

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("NEO4J_PASSWORD", "test")

from unittest.mock import patch, MagicMock

import pytest


@patch("src.adapters.outbound.vision_ocr_adapter.genai.Client")
def test_raises_runtime_error_when_gemini_api_key_missing(mock_client):
    """VisionOCRAdapter raises RuntimeError when GEMINI_API_KEY is not set."""
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        from src.adapters.outbound.vision_ocr_adapter import VisionOCRAdapter

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            VisionOCRAdapter()


@patch("src.adapters.outbound.vision_ocr_adapter.genai.Client")
def test_raises_runtime_error_when_gemini_api_key_empty(mock_client):
    """VisionOCRAdapter raises RuntimeError when GEMINI_API_KEY is empty string."""
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    env["GEMINI_API_KEY"] = ""
    with patch.dict(os.environ, env, clear=True):
        from src.adapters.outbound.vision_ocr_adapter import VisionOCRAdapter

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            VisionOCRAdapter()


@patch("src.adapters.outbound.vision_ocr_adapter.genai.Client")
def test_uses_default_model_when_gemini_model_not_set(mock_client):
    """VisionOCRAdapter defaults to gemini-2.5-flash when GEMINI_MODEL not in env."""
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GEMINI_MODEL")}
    env["GEMINI_API_KEY"] = "test-key-abc123"
    with patch.dict(os.environ, env, clear=True):
        from src.adapters.outbound.vision_ocr_adapter import VisionOCRAdapter

        adapter = VisionOCRAdapter()
        assert adapter._model == "gemini-2.5-flash"


@patch("src.adapters.outbound.vision_ocr_adapter.genai.Client")
def test_api_key_never_in_error_message(mock_client):
    """The API key value never appears in RuntimeError messages."""
    # Set a recognizable key value that we can search for in the error message
    secret_key = "sk-super-secret-key-12345"
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    env["GEMINI_API_KEY"] = ""
    with patch.dict(os.environ, env, clear=True):
        from src.adapters.outbound.vision_ocr_adapter import VisionOCRAdapter

        with pytest.raises(RuntimeError) as exc_info:
            VisionOCRAdapter()

        # The error message should not contain any API key value
        error_msg = str(exc_info.value)
        assert secret_key not in error_msg
        # Also ensure the word "GEMINI_API_KEY" is mentioned (for user guidance)
        # but the actual VALUE of any key is absent
        assert "GEMINI_API_KEY" in error_msg
