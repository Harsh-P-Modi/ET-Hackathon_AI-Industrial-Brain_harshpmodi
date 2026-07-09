"""Unit tests for classify_route keyword prefilter logic (task 3.1).

These tests validate that GRAPH_LOCAL and GRAPH_GLOBAL signal phrases are
detected case-insensitively, and that unmatched queries fall back to hybrid_fusion.
No Ollama daemon is needed — the health check is mocked out.
"""

from unittest.mock import patch, MagicMock

import pytest


def _make_adapter():
    """Create a LangGraphOrchestratorAdapter with mocked health check."""
    with patch("src.adapters.outbound.ollama_llm_adapter.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter
        from src.domain.services import HybridContextFuser

        # Minimal fakes for constructor dependencies.
        fake_vector = MagicMock()
        fake_graph = MagicMock()
        fake_fuser = MagicMock(spec=HybridContextFuser)

        adapter = LangGraphOrchestratorAdapter(
            vector_store=fake_vector,
            graph_store=fake_graph,
            fuser=fake_fuser,
        )
    return adapter


class TestClassifyRouteGraphLocal:
    """GRAPH_LOCAL signal phrases should return 'graph_local_search'."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = _make_adapter()

    @pytest.mark.parametrize(
        "query",
        [
            "What is downstream of pump P-101?",
            "Show upstream equipment for valve V-200",
            "Which equipment is connected to HX-301?",
            "What fails when compressor C-100 goes offline?",
            "List affected components after sensor failure",
            "This motor depends on a VFD",
        ],
    )
    def test_graph_local_phrases(self, query):
        assert self.adapter.classify_route(query) == "graph_local_search"

    def test_case_insensitive_graph_local(self):
        assert self.adapter.classify_route("DOWNSTREAM of P-101") == "graph_local_search"
        assert self.adapter.classify_route("Connected To HX-301") == "graph_local_search"
        assert self.adapter.classify_route("DEPENDS ON motor M-1") == "graph_local_search"


class TestClassifyRouteGraphGlobal:
    """GRAPH_GLOBAL signal phrases should return 'graph_global_search'."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = _make_adapter()

    @pytest.mark.parametrize(
        "query",
        [
            "What are common causes of bearing failure?",
            "Show patterns across all pumps",
            "What is the most frequent failure mode?",
            "Are there patterns across the plant?",
            "Give me a system-wide reliability overview",
        ],
    )
    def test_graph_global_phrases(self, query):
        assert self.adapter.classify_route(query) == "graph_global_search"

    def test_case_insensitive_graph_global(self):
        assert self.adapter.classify_route("COMMON CAUSES of vibration") == "graph_global_search"
        assert self.adapter.classify_route("ACROSS ALL units") == "graph_global_search"
        assert self.adapter.classify_route("System-Wide summary") == "graph_global_search"


class TestClassifyRouteFallback:
    """Queries without signal phrases should return 'hybrid_fusion'."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = _make_adapter()

    @pytest.mark.parametrize(
        "query",
        [
            "How do I replace a bearing?",
            "What is the maintenance schedule for P-101?",
            "Tell me about vibration analysis",
            "",
        ],
    )
    def test_no_keywords_returns_hybrid(self, query):
        assert self.adapter.classify_route(query) == "hybrid_fusion"
