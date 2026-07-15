"""
StreamlitChatAdapter — Jury-facing demo UI for FixMyPlant.

A pure HTTP client that communicates exclusively with the FastAPI backend.
Contains zero domain logic and zero imports from src/domain/ or src/ports/.
"""

import os

import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_BACKEND_URL = "http://localhost:8000"
HTTP_TIMEOUT_SECONDS = 30

ROUTE_BADGE_STYLES = {
    "vector_search": {"label": "Vector", "color": "#3498db"},
    "graph_local_search": {"label": "Graph-Local", "color": "#e67e22"},
    "graph_global_search": {"label": "Graph-Global", "color": "#9b59b6"},
}

ROUTE_BADGE_CSS = """
<style>
.route-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
}
.route-badge-vector { background-color: #3498db22; color: #3498db; border: 1px solid #3498db; }
.route-badge-graph-local { background-color: #e67e2222; color: #e67e22; border: 1px solid #e67e22; }
.route-badge-graph-global { background-color: #9b59b622; color: #9b59b6; border: 1px solid #9b59b6; }
</style>
"""

# TODO(pre-demo): Replace placeholder texts with final demo questions before jury presentation
DEMO_QUESTIONS = [
    "[PLACEHOLDER: Route A demo question — semantic/vector search]",
    "[PLACEHOLDER: Route B demo question — graph-local traversal]",
    "[PLACEHOLDER: Route C demo question — graph-global community summary]",
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


def send_query(question: str) -> dict | None:
    """Send question to backend. Returns parsed response or None on any failure."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question},
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
    # Map route key to CSS class suffix
    css_class_map = {
        "vector_search": "vector",
        "graph_local_search": "graph-local",
        "graph_global_search": "graph-global",
    }
    css_suffix = css_class_map.get(route_used, "vector")
    badge_html = f'<span class="route-badge route-badge-{css_suffix}">{style["label"]}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)


def render_citations(citations: list[dict]) -> None:
    """Render citations as an expandable section."""
    if not citations:
        return
    with st.expander(f"📚 Citations ({len(citations)})"):
        for citation in citations:
            source = citation.get("source_document", "Unknown source")
            snippet = citation.get("snippet", "")
            st.markdown(f"**{source}**")
            st.markdown(f"> {snippet}")
            st.divider()


def render_health_indicator(health: dict | None) -> None:
    """Display backend health status in the sidebar."""
    if health is None:
        st.sidebar.markdown("Neo4j: ❌  Ollama: ❌")
        st.sidebar.caption("Backend unreachable")
        return
    neo4j_icon = "✅" if health.get("neo4j") else "❌"
    ollama_icon = "✅" if health.get("ollama") else "❌"
    st.sidebar.markdown(f"Neo4j: {neo4j_icon}  Ollama: {ollama_icon}")


def render_demo_buttons() -> str | None:
    """Render pre-loaded demo question buttons. Returns selected question or None."""
    cols = st.columns(len(DEMO_QUESTIONS))
    for i, (col, question) in enumerate(zip(cols, DEMO_QUESTIONS)):
        with col:
            if st.button(f"Demo {i + 1}", key=f"demo_btn_{i}", help=question):
                return question
    return None


# ─── Main App Flow ────────────────────────────────────────────────────────────


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="FixMyPlant — AI Industrial Brain",
        page_icon="🏥",
        layout="wide",
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Inject custom CSS
    inject_custom_css()

    # Sidebar — health indicator and file uploader
    with st.sidebar:
        st.title("🏥 FixMyPlant")
        st.markdown("---")
        st.subheader("Backend Health")
        health = fetch_health()
        render_health_indicator(health)
        st.markdown("---")
        st.subheader("📄 Document Ingestion")
        st.file_uploader(
            "Upload a document",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            key="file_uploader",
        )

        uploaded_file = st.session_state.get("file_uploader")
        if uploaded_file is not None:
            result = send_ingest(uploaded_file.getvalue(), uploaded_file.name)
            if result is not None:
                st.sidebar.success(f"✅ {uploaded_file.name} accepted for ingestion")
            else:
                st.sidebar.error("Could not upload file. Please ensure the backend is running.")

    # Main area — Chat title
    st.title("💬 Ask FixMyPlant")

    # Demo question buttons
    demo_question = render_demo_buttons()

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
            with st.spinner("Thinking..."):
                result = send_query(question)

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
                error_msg = "I'm sorry, I couldn't get a response. Please ensure the backend service is running and try again."
                st.markdown(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                    "route_used": None,
                    "citations": None,
                })


if __name__ == "__main__":
    main()
