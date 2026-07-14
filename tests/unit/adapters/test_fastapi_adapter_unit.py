"""Unit tests for FastAPIAdapter constructor and import purity."""

import ast

import pytest
from pathlib import Path
from unittest.mock import Mock

from src.adapters.inbound.fastapi_adapter import FastAPIAdapter


@pytest.fixture
def mock_adapter():
    """Create FastAPIAdapter with all mock dependencies."""
    adapter = FastAPIAdapter(
        query_service=Mock(),
        ingestion_service=Mock(),
        vector_store=Mock(),
        graph_store=Mock(),
        llm_inference=Mock(),
        neo4j_ping=lambda: True,
        ollama_ping=lambda: True,
    )
    return adapter


class TestConstructor:
    """Tests for FastAPIAdapter constructor with mock dependencies."""

    def test_constructor_succeeds_with_all_dependencies(self, mock_adapter):
        """Constructor with all valid mock dependencies succeeds."""
        assert mock_adapter.app is not None

    def test_app_is_fastapi_instance(self, mock_adapter):
        """The app property returns a FastAPI instance."""
        from fastapi import FastAPI

        assert isinstance(mock_adapter.app, FastAPI)


class TestImportPurity:
    """Verify no banned vendor SDK imports in the adapter file."""

    BANNED_MODULES = {"neo4j", "ollama", "langgraph", "google", "anthropic", "openai"}

    def test_no_banned_imports_via_ast(self):
        """No banned imports (neo4j, ollama, langgraph, etc.) in fastapi_adapter.py via AST inspection."""
        adapter_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "adapters"
            / "inbound"
            / "fastapi_adapter.py"
        )
        source = adapter_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split(".")[0])

        violations = imported_modules & self.BANNED_MODULES
        assert not violations, f"Banned imports found: {violations}"


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""

    def test_cors_no_wildcard_origin(self, mock_adapter):
        """Verify CORS middleware does not use wildcard '*' origin."""
        from starlette.middleware.cors import CORSMiddleware

        app = mock_adapter.app
        # Access the middleware stack
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                origins = middleware.kwargs.get("allow_origins", [])
                assert "*" not in origins, "CORS must not use wildcard origin"
                assert "http://localhost:8501" in origins
                assert "http://127.0.0.1:8501" in origins
                return
        pytest.fail("CORSMiddleware not found in middleware stack")
