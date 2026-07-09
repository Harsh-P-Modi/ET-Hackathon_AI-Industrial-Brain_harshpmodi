"""LangGraphOrchestratorAdapter — implements LLMInferencePort via local Ollama HTTP API.

This adapter encapsulates all Ollama daemon interaction and houses a LangGraph
state machine for query orchestration. It exposes classify_route, synthesize,
and embed to the domain layer.

Only adapter-level code lives here; no domain logic.
"""

import hashlib
import json
import re
from typing import TYPE_CHECKING, TypedDict, Union

import requests
from langgraph.graph import StateGraph, END

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from src.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_EMBED_MODEL, VECTOR_DIMENSIONS
from src.domain.entities import DocumentChunk, EquipmentNode, MaintenanceEvent
from src.domain.services import HybridContextFuser
from src.ports.outbound import GraphStoragePort, VectorStoragePort


class OllamaAdapterError(Exception):
    """Raised when an Ollama HTTP interaction fails unrecoverably.

    Carries structured diagnostic information: operation name, endpoint URL,
    and error detail from the underlying failure.
    """

    def __init__(self, operation: str, endpoint: str, detail: str) -> None:
        self.operation = operation
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(
            f"OllamaAdapterError: {operation} failed at {endpoint}: {detail}"
        )


class OrchestratorState(TypedDict):
    """LangGraph state schema for query orchestration pipeline."""

    query: str
    route: str
    vector_results: list[DocumentChunk]
    graph_results: list[Union[EquipmentNode, MaintenanceEvent]]
    fused_context: list[Union[DocumentChunk, EquipmentNode, MaintenanceEvent]]
    answer: str


class LangGraphOrchestratorAdapter:
    """Outbound adapter implementing LLMInferencePort against a local Ollama daemon.

    Constructor performs a health check (GET /api/tags) and fails fast with a
    RuntimeError if the daemon is unreachable.
    """

    def __init__(
        self,
        vector_store: VectorStoragePort,
        graph_store: GraphStoragePort,
        fuser: HybridContextFuser,
    ) -> None:
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._fuser = fuser

        self._host = OLLAMA_HOST
        self._model = OLLAMA_MODEL
        self._embed_model = OLLAMA_EMBED_MODEL

        # SHA-256 keyed embedding cache — instance-scoped, no TTL/eviction.
        self._embedding_cache: dict[str, list[float]] = {}

        # Health check: fail fast if Ollama daemon is unreachable.
        self._health_check()

    def _health_check(self) -> None:
        """GET /api/tags to verify Ollama daemon is reachable."""
        endpoint = f"{self._host}/api/tags"
        try:
            response = requests.get(endpoint, timeout=5)
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama daemon is not reachable at {self._host}: {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama daemon timed out at {endpoint}: {exc}"
            ) from exc
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Ollama daemon returned an error at {endpoint}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # LLMInferencePort interface methods (stubs for now)
    # ------------------------------------------------------------------

    # Signal phrases for keyword-based route classification.
    _GRAPH_LOCAL_PHRASES: tuple[str, ...] = (
        "downstream",
        "upstream",
        "connected to",
        "fails",
        "affected",
        "depends on",
    )
    _GRAPH_GLOBAL_PHRASES: tuple[str, ...] = (
        "common causes",
        "across all",
        "most frequent",
        "patterns across",
        "system-wide",
    )

    # Valid route values from RouteType enum.
    _VALID_ROUTES: frozenset[str] = frozenset({
        "vector_search",
        "graph_local_search",
        "graph_global_search",
        "hybrid_fusion",
    })

    _CLASSIFY_SYSTEM_PROMPT: str = (
        "You are a query classifier for an industrial knowledge system. "
        "Classify the query into exactly one of: vector_search, "
        "graph_local_search, graph_global_search, hybrid_fusion. "
        "Return JSON with a single key 'route'."
    )

    def classify_route(self, query: str) -> str:
        """Classify a query into a retrieval route.

        Uses a cheap keyword prefilter first. If any GRAPH_LOCAL or GRAPH_GLOBAL
        signal phrase is detected (case-insensitive), the route is returned
        immediately without invoking the Ollama daemon. Otherwise falls back to
        LLM classification via Ollama's /api/generate endpoint with JSON mode.

        On any error (network, invalid JSON, invalid route value), returns
        "hybrid_fusion" as a safe default.
        """
        query_lower = query.lower()

        # Check GRAPH_LOCAL keywords first.
        for phrase in self._GRAPH_LOCAL_PHRASES:
            if phrase in query_lower:
                return "graph_local_search"

        # Check GRAPH_GLOBAL keywords.
        for phrase in self._GRAPH_GLOBAL_PHRASES:
            if phrase in query_lower:
                return "graph_global_search"

        # No keyword match — invoke LLM fallback classification.
        endpoint = f"{self._host}/api/generate"
        payload = {
            "model": self._model,
            "prompt": query,
            "system": self._CLASSIFY_SYSTEM_PROMPT,
            "format": "json",
            "stream": False,
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            outer = response.json()
            inner_json = json.loads(outer["response"])
            route = inner_json["route"]
            if route in self._VALID_ROUTES:
                return route
            # Valid JSON but invalid route value — fall back.
            raise OllamaAdapterError(
                operation="classify_route",
                endpoint=endpoint,
                detail=f"Invalid route value: {route!r}",
            )
        except OllamaAdapterError:
            # Log/wrap occurred above; graceful degradation.
            return "hybrid_fusion"
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.HTTPError,
        ) as exc:
            # Network-level failure — wrap for diagnostics, return fallback.
            OllamaAdapterError(
                operation="classify_route",
                endpoint=endpoint,
                detail=str(exc),
            )
            return "hybrid_fusion"
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # Malformed or unparseable JSON response — wrap, return fallback.
            OllamaAdapterError(
                operation="classify_route",
                endpoint=endpoint,
                detail=f"Invalid response JSON: {exc}",
            )
            return "hybrid_fusion"

    def _serialize_context(
        self, items: list[DocumentChunk | EquipmentNode | MaintenanceEvent]
    ) -> str:
        """Serialize context items into a tagged text block for the LLM prompt.

        Each item is rendered with a type-specific source tag and separated by
        a blank line. Domain identifiers are preserved in the output for
        downstream citation extraction.
        """
        parts: list[str] = []
        for item in items:
            if isinstance(item, DocumentChunk):
                parts.append(
                    f"[chunk_{item.chunk_id}] (from: {item.source_document})\n"
                    f"{item.text}"
                )
            elif isinstance(item, EquipmentNode):
                connects = ", ".join(item.connects_to)
                parts.append(
                    f"[equip_{item.equipment_id}] (type: {item.equipment_type})\n"
                    f"Name: {item.name}, Connects to: {connects}"
                )
            elif isinstance(item, MaintenanceEvent):
                parts.append(
                    f"[event_{item.event_id}] (equipment: {item.equipment_id}, "
                    f"date: {item.timestamp.isoformat()})\n"
                    f"{item.description}"
                )
        return "\n\n".join(parts)

    def synthesize(
        self,
        query: str,
        context: list[DocumentChunk | EquipmentNode | MaintenanceEvent],
    ) -> str:
        """Synthesize an answer from query and retrieved context.

        Builds a grounded prompt from serialized context items and the user
        query, then calls Ollama's /api/generate endpoint. When 2+ context
        items are provided, the system prompt includes citation instructions.

        Raises:
            OllamaAdapterError: On network errors, HTTP errors, missing or
                empty response from the Ollama daemon.
        """
        serialized_context = self._serialize_context(context)

        # Build system prompt — always enforce grounded-only instruction.
        system_prompt = (
            "Answer ONLY from the provided context. "
            "Do not use your own training data."
        )
        # When multiple context items exist, add citation instruction.
        if len(context) >= 2:
            system_prompt += (
                " Cite sources inline using [chunk_XXX], [equip_XXX], "
                "or [event_XXX] tags."
            )

        user_prompt = f"Context:\n{serialized_context}\n\nQuestion: {query}"

        endpoint = f"{self._host}/api/generate"
        payload = {
            "model": self._model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            raise OllamaAdapterError(
                operation="synthesize",
                endpoint=endpoint,
                detail=str(exc),
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaAdapterError(
                operation="synthesize",
                endpoint=endpoint,
                detail=f"Failed to parse response JSON: {exc}",
            ) from exc

        answer = data.get("response")
        if not answer:
            raise OllamaAdapterError(
                operation="synthesize",
                endpoint=endpoint,
                detail=(
                    "Empty or missing 'response' field in Ollama reply. "
                    f"Keys: {list(data.keys())}"
                ),
            )

        return answer

    def embed(self, text: str) -> list[float]:
        """Produce a 768-dimensional embedding for the given text.

        Uses SHA-256 hash of the input text as a cache key. On cache hit,
        returns the stored embedding immediately. On cache miss, calls the
        Ollama /api/embeddings endpoint and validates the response dimension.

        Raises:
            ValueError: If the returned embedding has incorrect dimensionality.
            OllamaAdapterError: On network/timeout/HTTP errors or missing
                embedding field in the response.
        """
        cache_key = hashlib.sha256(text.encode()).hexdigest()

        # Cache hit — return immediately.
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Cache miss — call Ollama.
        endpoint = f"{self._host}/api/embeddings"
        payload = {"model": self._embed_model, "prompt": text}

        try:
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            raise OllamaAdapterError(
                operation="embed",
                endpoint=endpoint,
                detail=str(exc),
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaAdapterError(
                operation="embed",
                endpoint=endpoint,
                detail=f"Failed to parse response JSON: {exc}",
            ) from exc

        if "embedding" not in data:
            raise OllamaAdapterError(
                operation="embed",
                endpoint=endpoint,
                detail=f"Response missing 'embedding' field. Keys: {list(data.keys())}",
            )

        embedding: list[float] = data["embedding"]

        if len(embedding) != VECTOR_DIMENSIONS:
            raise ValueError(
                f"Embedding dimension mismatch: expected {VECTOR_DIMENSIONS}, "
                f"got {len(embedding)}"
            )

        # Store in cache before returning.
        self._embedding_cache[cache_key] = embedding
        return embedding

    # ------------------------------------------------------------------
    # Internal tools for LangGraph state machine (not on LLMInferencePort)
    # ------------------------------------------------------------------

    def _vector_retrieval_tool(self, query: str) -> list[DocumentChunk]:
        """Internal tool: embed query then semantic search."""
        embedding = self.embed(query)
        return self._vector_store.semantic_search(embedding, top_k=10)

    def _graph_cypher_tool(
        self, equipment_id: str, depth: int = 2
    ) -> list[EquipmentNode]:
        """Internal tool: graph neighbor traversal."""
        return self._graph_store.get_neighbors(equipment_id, depth)

    # ------------------------------------------------------------------
    # LangGraph state machine nodes and routing
    # ------------------------------------------------------------------

    def _classify_node(self, state: OrchestratorState) -> dict:
        """Classify the query and store the route in state."""
        route = self.classify_route(state["query"])
        return {"route": route}

    def _vector_retrieve_node(self, state: OrchestratorState) -> dict:
        """Perform vector retrieval and store results in state."""
        results = self._vector_retrieval_tool(state["query"])
        return {"vector_results": results}

    def _graph_retrieve_node(self, state: OrchestratorState) -> dict:
        """Perform graph retrieval and store results in state.

        Extracts equipment IDs from the query using a simple regex pattern,
        then calls graph traversal for each found ID. If no equipment IDs
        are found, returns an empty list.
        """
        query = state["query"]
        equipment_ids = re.findall(r"\b[A-Za-z]+\-[0-9]+\b", query)

        all_results: list[Union[EquipmentNode, MaintenanceEvent]] = []
        for eq_id in equipment_ids:
            neighbors = self._graph_cypher_tool(eq_id, depth=2)
            all_results.extend(neighbors)

        # If no equipment IDs found, attempt a broad search with a default ID
        # (graceful degradation — return empty if nothing to search)
        return {"graph_results": all_results}

    def _fuse_node(self, state: OrchestratorState) -> dict:
        """Fuse vector and graph results via domain HybridContextFuser.

        This is a post-hoc domain call — NOT a LangGraph tool. Handles the
        case where only one type of results exists by passing empty list for
        the missing type.
        """
        vector_results = state.get("vector_results") or []
        graph_results = state.get("graph_results") or []
        fused = self._fuser.fuse(vector_results, graph_results)
        return {"fused_context": fused}

    def _synthesize_node(self, state: OrchestratorState) -> dict:
        """Synthesize an answer from the fused context and store it in state."""
        query = state["query"]
        fused_context = state.get("fused_context") or []
        answer = self.synthesize(query, fused_context)
        return {"answer": answer}

    def _route_after_classify(self, state: OrchestratorState) -> str:
        """Conditional routing: returns the next node name based on route.

        - VECTOR or HYBRID → "vector_retrieve" (HYBRID continues to graph after)
        - GRAPH_LOCAL or GRAPH_GLOBAL → "graph_retrieve"
        """
        route = state["route"]
        if route in ("vector_search", "hybrid_fusion"):
            return "vector_retrieve"
        elif route in ("graph_local_search", "graph_global_search"):
            return "graph_retrieve"
        # Fallback to vector_retrieve for any unexpected route value
        return "vector_retrieve"

    def _route_after_vector_retrieve(self, state: OrchestratorState) -> str:
        """Conditional routing after vector retrieval.

        If HYBRID, continue to graph_retrieve before fusing.
        Otherwise, go directly to fuse.
        """
        if state["route"] == "hybrid_fusion":
            return "graph_retrieve"
        return "fuse"

    def _build_graph(self) -> "CompiledStateGraph":
        """Build and compile the LangGraph state machine for query orchestration.

        Flow:
        - VECTOR: classify → vector_retrieve → fuse → synthesize → END
        - GRAPH_LOCAL/GRAPH_GLOBAL: classify → graph_retrieve → fuse → synthesize → END
        - HYBRID: classify → vector_retrieve → graph_retrieve → fuse → synthesize → END
        """
        graph = StateGraph(OrchestratorState)

        # Add nodes
        graph.add_node("classify", self._classify_node)
        graph.add_node("vector_retrieve", self._vector_retrieve_node)
        graph.add_node("graph_retrieve", self._graph_retrieve_node)
        graph.add_node("fuse", self._fuse_node)
        graph.add_node("synthesize", self._synthesize_node)

        # Entry point
        graph.set_entry_point("classify")

        # Conditional edges after classify
        graph.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "vector_retrieve": "vector_retrieve",
                "graph_retrieve": "graph_retrieve",
            },
        )

        # Conditional edges after vector_retrieve (HYBRID goes to graph, others to fuse)
        graph.add_conditional_edges(
            "vector_retrieve",
            self._route_after_vector_retrieve,
            {
                "graph_retrieve": "graph_retrieve",
                "fuse": "fuse",
            },
        )

        # graph_retrieve always goes to fuse
        graph.add_edge("graph_retrieve", "fuse")

        # fuse always goes to synthesize
        graph.add_edge("fuse", "synthesize")

        # synthesize goes to END
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _run_orchestration(self, query: str) -> str:
        """Run the full orchestration pipeline for a query.

        Compiles the LangGraph state machine, invokes it with the query,
        and returns the final answer string.

        Raises:
            OllamaAdapterError: If any node in the pipeline fails.
        """
        try:
            compiled_graph = self._build_graph()
            initial_state: OrchestratorState = {
                "query": query,
                "route": "",
                "vector_results": [],
                "graph_results": [],
                "fused_context": [],
                "answer": "",
            }
            final_state = compiled_graph.invoke(initial_state)
            return final_state["answer"]
        except OllamaAdapterError:
            raise
        except Exception as exc:
            raise OllamaAdapterError(
                operation="_run_orchestration",
                endpoint=f"{self._host}/api/generate",
                detail=f"Orchestration pipeline failed: {exc}",
            ) from exc
