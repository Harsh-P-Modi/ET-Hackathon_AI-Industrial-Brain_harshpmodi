"""
StreamlitChatAdapter — Jury-facing demo UI for FixMyPlant.

A pure HTTP client that communicates exclusively with the FastAPI backend.
Contains zero domain logic and zero imports from src/domain/ or src/ports/.
"""

import os
from pathlib import Path

import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_BACKEND_URL = "http://localhost:8000"
HTTP_TIMEOUT_SECONDS = 300

ROUTE_BADGE_STYLES = {
    "vector_search": {"label": "🔍 Vector Search", "color": "#3498db"},
    "graph_local_search": {"label": "🔗 Graph-Local", "color": "#e67e22"},
    "graph_global_search": {"label": "🌐 Graph-Global", "color": "#9b59b6"},
    "hybrid_fusion": {"label": "⚡ Hybrid Fusion", "color": "#2ecc71"},
}

ROUTE_BADGE_CSS = """
<style>
.route-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 4px;
}
.route-badge-vector { background-color: #3498db22; color: #3498db; border: 1px solid #3498db; }
.route-badge-graph-local { background-color: #e67e2222; color: #e67e22; border: 1px solid #e67e22; }
.route-badge-graph-global { background-color: #9b59b622; color: #9b59b6; border: 1px solid #9b59b6; }
.route-badge-hybrid { background-color: #2ecc7122; color: #2ecc71; border: 1px solid #2ecc71; }

/* Header styling */
.main-header {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.main-header h1 {
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

/* Route info cards */
.route-card {
    background: #1a1f2e;
    border-radius: 8px;
    padding: 12px 16px;
    border-left: 3px solid;
    margin-bottom: 8px;
}
.route-card-vector { border-left-color: #3498db; }
.route-card-graph { border-left-color: #e67e22; }
.route-card-hybrid { border-left-color: #2ecc71; }

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot-green { background-color: #2ecc71; }
.status-dot-red { background-color: #e74c3c; }
</style>
"""

DEMO_QUESTIONS = [
    "What are the water quality requirements for the boiler feedwater system?",
    "If pump CW-P-101 fails, what downstream equipment is affected?",
    "What are the most common causes of unplanned shutdowns across all systems?",
]

DEMO_DESCRIPTIONS = [
    "Tests **Vector Search** — finds relevant document chunks by semantic similarity",
    "Tests **Graph Traversal** — follows equipment connections in the knowledge graph",
    "Tests **Graph Global** — aggregates patterns across community summaries",
]


def get_backend_url() -> str:
    """Resolve BACKEND_URL with priority: env var > st.secrets > default."""
    # 1. Environment variable (highest priority)
    env_url = os.environ.get("BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")

    # 2. Streamlit secrets
    try:
        import streamlit as st

        secret_url = st.secrets.get("BACKEND_URL")
        if secret_url:
            return str(secret_url).rstrip("/")
    except (FileNotFoundError, KeyError):
        pass

    # 3. Default
    return DEFAULT_BACKEND_URL


# Resolve once at module load
BACKEND_URL = get_backend_url()


# ─── HTTP Client Functions ────────────────────────────────────────────────────


def send_query(question: str, history: list[dict] | None = None) -> dict | None:
    """Send question to backend with optional conversation history."""
    try:
        payload = {"question": question}
        if history:
            # Send last 4 messages (2 exchanges) for context
            payload["history"] = [
                {"role": m["role"], "content": m["content"][:200]}
                for m in history[-4:]
            ]
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None
    except requests.Timeout:
        return None
    except requests.HTTPError:
        return None
    except Exception:
        return None


def send_ingest(file_bytes: bytes, filename: str) -> dict | None:
    """Upload file to backend for ingestion. Returns parsed response or None on any failure."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/ingest",
            files={"file": (filename, file_bytes)},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None
    except requests.Timeout:
        return None
    except requests.HTTPError:
        return None
    except Exception:
        return None


def fetch_health() -> dict | None:
    """Check backend health. Returns parsed response or None on any failure."""
    try:
        resp = requests.get(
            f"{BACKEND_URL}/health",
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None
    except requests.Timeout:
        return None
    except requests.HTTPError:
        return None
    except Exception:
        return None


# ─── UI Rendering Functions ───────────────────────────────────────────────────


def inject_custom_css() -> None:
    """Inject custom CSS styles for route badges and other UI elements."""
    st.markdown(ROUTE_BADGE_CSS, unsafe_allow_html=True)


def render_route_badge(route_used: str) -> None:
    """Render a colored badge indicating which retrieval route was used."""
    style = ROUTE_BADGE_STYLES.get(route_used)
    if style is None:
        return
    css_class_map = {
        "vector_search": "vector",
        "graph_local_search": "graph-local",
        "graph_global_search": "graph-global",
        "hybrid_fusion": "hybrid",
    }
    css_suffix = css_class_map.get(route_used, "vector")
    badge_html = f'<span class="route-badge route-badge-{css_suffix}">{style["label"]}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)


def render_citations(citations: list[dict]) -> None:
    """Render citations as an expandable section."""
    if not citations:
        return
    with st.expander(f"📚 Sources ({len(citations)})"):
        for citation in citations:
            source = citation.get("source_document", "Unknown source")
            snippet = citation.get("snippet", "")
            # Clean up the source name for display
            display_name = source.replace("_", " ").replace(".txt", "").title()
            st.markdown(f"**📄 {display_name}**")
            st.caption(f"File: {source}")
            if snippet:
                # Show a meaningful excerpt
                st.markdown(f"> {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
            st.divider()


def render_health_indicator(health: dict | None) -> None:
    """Display backend health status in the sidebar."""
    if health is None:
        st.markdown(
            '<span class="status-dot status-dot-red"></span>Backend unreachable',
            unsafe_allow_html=True,
        )
        return
    neo4j_ok = health.get("neo4j", False)
    ollama_ok = health.get("ollama", False)
    neo4j_dot = "status-dot-green" if neo4j_ok else "status-dot-red"
    ollama_dot = "status-dot-green" if ollama_ok else "status-dot-red"
    st.markdown(
        f'<span class="status-dot {neo4j_dot}"></span>Neo4j &nbsp;&nbsp;'
        f'<span class="status-dot {ollama_dot}"></span>Ollama',
        unsafe_allow_html=True,
    )


def render_demo_buttons() -> str | None:
    """Render pre-loaded demo question buttons with descriptions."""
    selected = None
    for i, (question, desc) in enumerate(zip(DEMO_QUESTIONS, DEMO_DESCRIPTIONS)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"_{desc}_")
        with col2:
            if st.button(f"▶ Demo {i + 1}", key=f"demo_btn_{i}", use_container_width=True):
                selected = question
    return selected


def get_dataset_dir() -> Path:
    """Resolve the golden dataset directory path."""
    # Try relative to this file first (src/adapters/inbound/ → project root)
    adapter_dir = Path(__file__).resolve().parent
    project_root = adapter_dir.parent.parent.parent
    dataset_dir = project_root / "data" / "golden_dataset"
    if dataset_dir.is_dir():
        return dataset_dir
    # Fallback: try CWD-based
    cwd_dataset = Path.cwd() / "data" / "golden_dataset"
    if cwd_dataset.is_dir():
        return cwd_dataset
    return dataset_dir  # return even if missing, caller checks


def render_knowledge_base() -> None:
    """Render the knowledge base section showing ingested documents."""
    dataset_dir = get_dataset_dir()

    if not dataset_dir.is_dir():
        st.caption("No documents found.")
        return

    files = sorted(dataset_dir.iterdir())
    doc_files = [f for f in files if f.is_file()]

    if not doc_files:
        st.caption("No documents found.")
        return

    st.caption(f"{len(doc_files)} documents loaded")

    for doc_file in doc_files:
        file_size_kb = doc_file.stat().st_size / 1024
        ext = doc_file.suffix.upper().lstrip(".")

        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                # Show file info
                st.markdown(f"📄 **{doc_file.stem}**")
                st.caption(f"{ext} · {file_size_kb:.1f} KB")
            with col2:
                # Download button
                file_bytes = doc_file.read_bytes()
                st.download_button(
                    label="⬇️",
                    data=file_bytes,
                    file_name=doc_file.name,
                    mime="text/plain" if ext == "TXT" else "application/octet-stream",
                    key=f"dl_{doc_file.name}",
                )
            with col3:
                # Delete button
                if st.button("🗑️", key=f"del_{doc_file.name}", help=f"Delete {doc_file.name}"):
                    doc_file.unlink()
                    st.rerun()

    # View document content in expander
    with st.expander("👁️ Preview Documents"):
        selected_doc = st.selectbox(
            "Select document",
            options=[f.name for f in doc_files],
            key="doc_preview_select",
        )
        if selected_doc:
            doc_path = dataset_dir / selected_doc
            try:
                content = doc_path.read_text(encoding="utf-8")
                st.code(content, language=None)
            except Exception:
                st.warning("Cannot preview this file type.")


# ─── Graph Visualization ──────────────────────────────────────────────────────


def fetch_graph_data() -> dict | None:
    """Fetch full graph data from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/graph", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def render_graph_visualization() -> None:
    """Render interactive knowledge graph using pyvis (full HTML embed)."""
    import streamlit.components.v1 as components
    from pyvis.network import Network
    import tempfile

    st.markdown("### 🕸️ Equipment Knowledge Graph")

    graph_data = fetch_graph_data()
    if graph_data is None or not graph_data.get("nodes"):
        st.info("Graph data not available. Ensure the backend is running with data ingested.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    # Get unique types for filtering
    all_types = sorted(set(n.get("type", "unknown") for n in graph_data["nodes"]))

    with col1:
        selected_types = st.multiselect(
            "Filter by equipment type",
            options=all_types,
            default=all_types,
            key="graph_type_filter",
        )
    with col2:
        show_labels = st.checkbox("Show labels", value=True, key="graph_labels")
    with col3:
        physics_enabled = st.checkbox("Physics simulation", value=True, key="graph_physics")

    # Color map for equipment types
    type_colors = {
        "pump": "#3498db",
        "heat_exchanger": "#e74c3c",
        "tank": "#2ecc71",
        "valve": "#f39c12",
        "compressor": "#9b59b6",
        "boiler": "#e67e22",
        "sensor": "#1abc9c",
        "flow_control_valve": "#f39c12",
        "temperature_control_valve": "#f39c12",
        "flow_transmitter": "#1abc9c",
        "temperature_transmitter": "#1abc9c",
        "pressure_transmitter": "#1abc9c",
        "generator": "#8e44ad",
        "steam_turbine": "#c0392b",
        "fire_pump": "#e74c3c",
        "gas_detector": "#16a085",
        "flame_detector": "#d35400",
        "cooling_water": "#3498db",
        "boiler_feedwater": "#2980b9",
        "equipment": "#95a5a6",
    }

    # Filter nodes
    filtered_node_ids = set()
    for n in graph_data["nodes"]:
        if n.get("type", "unknown") in selected_types:
            filtered_node_ids.add(n["id"])

    # Build pyvis network
    net = Network(
        height="550px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="#e6edf3",
    )

    # Configure physics
    if physics_enabled:
        net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150)
    else:
        net.toggle_physics(False)

    # Add nodes
    for n in graph_data["nodes"]:
        if n["id"] not in filtered_node_ids:
            continue
        color = type_colors.get(n.get("type", ""), "#95a5a6")
        label = n.get("label", n["id"]) if show_labels else ""
        net.add_node(
            n["id"],
            label=label,
            title=f"<b>{n.get('label', n['id'])}</b><br>Type: {n.get('type', 'unknown')}<br>ID: {n['id']}",
            color=color,
            size=22,
            borderWidth=2,
            borderWidthSelected=4,
        )

    # Add edges (only between visible nodes)
    for e in graph_data["edges"]:
        if e["from"] in filtered_node_ids and e["to"] in filtered_node_ids:
            net.add_edge(
                e["from"],
                e["to"],
                color="#666",
                arrows="to",
                width=1.5,
            )

    # Configure interaction
    net.set_options("""
    {
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": {
                "enabled": true
            },
            "zoomView": true,
            "dragView": true
        },
        "nodes": {
            "shape": "dot",
            "font": {
                "size": 11,
                "color": "#e6edf3"
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            }
        }
    }
    """)

    # Render to HTML and embed
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html_content = open(f.name).read()

    components.html(html_content, height=580, scrolling=False)

    # Legend + stats
    st.markdown("---")
    legend_col1, legend_col2 = st.columns(2)
    with legend_col1:
        st.markdown("**Legend:**")
        legend_items = [
            ("🔵 Pump", "pump"),
            ("🔴 Heat Exchanger", "heat_exchanger"),
            ("🟢 Tank", "tank"),
            ("🟠 Valve", "valve"),
            ("🟣 Compressor", "compressor"),
        ]
        for label, _ in legend_items:
            st.caption(label)
    with legend_col2:
        st.markdown("**Stats:**")
        st.caption(f"Nodes shown: {len(filtered_node_ids)}")
        st.caption(f"Total nodes: {len(graph_data['nodes'])}")
        st.caption(f"Total connections: {len(graph_data['edges'])}")
        st.caption("💡 Scroll to zoom · Drag to pan · Click node to highlight")


# ─── Main App Flow ────────────────────────────────────────────────────────────


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="FixMyPlant — AI Industrial Brain",
        page_icon="🏭",
        layout="wide",
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Inject custom CSS
    inject_custom_css()

    # ─── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏭 FixMyPlant")
        st.caption("AI Industrial Brain")
        st.markdown("---")

        # System status
        st.markdown("**System Status**")
        health = fetch_health()
        render_health_indicator(health)
        st.markdown("---")

        # Knowledge Base
        st.markdown("**📚 Knowledge Base**")
        render_knowledge_base()
        st.markdown("---")

        # Document upload
        st.markdown("**➕ Add Document**")
        st.file_uploader(
            "Upload a document",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "txt"],
            key="file_uploader",
            label_visibility="collapsed",
        )

        uploaded_file = st.session_state.get("file_uploader")
        if uploaded_file is not None:
            result = send_ingest(uploaded_file.getvalue(), uploaded_file.name)
            if result is not None:
                st.success(f"✅ {uploaded_file.name} ingested")
            else:
                st.error("Upload failed — backend unreachable")

        st.markdown("---")
        st.caption("Powered by Ollama · Neo4j · LangGraph")

    # ─── Main Area ────────────────────────────────────────────────────────────

    # Header
    st.markdown(
        '<div class="main-header">'
        "<h1>🏭 FixMyPlant</h1>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle" style="text-align:center;">'
        "Ask questions about your industrial equipment — powered by hybrid RAG with knowledge graph intelligence"
        "</p>",
        unsafe_allow_html=True,
    )

    # Route explanation (collapsed by default)
    with st.expander("ℹ️ How it works — Retrieval Routes", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="route-card route-card-vector">'
                "<strong>🔍 Vector Search</strong><br>"
                "<small>Semantic similarity over document chunks. "
                'Best for: "What are the specs for...?"</small>'
                "</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="route-card route-card-graph">'
                "<strong>🔗 Graph Traversal</strong><br>"
                "<small>Follows equipment connections in the knowledge graph. "
                'Best for: "If X fails, what\'s affected?"</small>'
                "</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="route-card route-card-hybrid">'
                "<strong>⚡ Hybrid Fusion</strong><br>"
                "<small>Combines vector + graph results via Reciprocal Rank Fusion. "
                "Used for complex queries.</small>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Tabs: Chat and Graph Visualization
    tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸️ Knowledge Graph"])

    with tab_graph:
        render_graph_visualization()

    with tab_chat:
        # Demo question buttons
        st.markdown("**Try a demo question:**")
        demo_question = render_demo_buttons()
        st.markdown("---")

        # Display chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("route_used"):
                        render_route_badge(msg["route_used"])
                    if msg.get("citations"):
                        render_citations(msg["citations"])

    # Handle new input (from chat input or demo button)
    user_input = st.chat_input("Ask a question about your equipment...")
    question = demo_question or user_input

    if question:
        # Append and display user message
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Query backend with spinner
        with st.chat_message("assistant"):
            with st.spinner("Analyzing with hybrid RAG pipeline..."):
                result = send_query(question, history=st.session_state["messages"])

            if result is not None:
                answer = result.get("answer", "")
                route_used = result.get("route_used")
                citations = result.get("citations", [])

                st.markdown(answer)
                if route_used:
                    render_route_badge(route_used)
                if citations:
                    render_citations(citations)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "route_used": route_used,
                    "citations": citations,
                })
            else:
                error_msg = "⚠️ Could not reach the backend. Please ensure the service is running."
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                    "route_used": None,
                    "citations": None,
                })


if __name__ == "__main__":
    main()
