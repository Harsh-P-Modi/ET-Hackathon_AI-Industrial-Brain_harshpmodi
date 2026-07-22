# 🏭 FixMyPlant — AI Industrial Brain

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Neo4j 5](https://img.shields.io/badge/graph-Neo4j%205-008CC1.svg)](https://neo4j.com/)
[![Docker](https://img.shields.io/badge/deploy-Docker%20Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

**FixMyPlant** is a hybrid Retrieval-Augmented Generation (RAG) system for industrial plant maintenance. It answers natural-language questions about equipment by combining semantic search over documents with graph traversal over equipment topology — grounded in P&ID diagrams, maintenance logs, and safety manuals, with full source citations.

> Built with a hexagonal (ports & adapters) architecture, running entirely on self-hosted, open-source infrastructure (Neo4j + Ollama). No data leaves your machine unless Gemini Vision OCR is explicitly enabled for document ingestion.

---

## Table of Contents

- [Why FixMyPlant](#why-fixmyplant)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Option A — Docker (recommended)](#option-a--docker-recommended)
  - [Option B — Local Python](#option-b--local-python)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Operating the Docker Stack](#operating-the-docker-stack)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why FixMyPlant

Plant maintenance knowledge is scattered across PDFs, scanned P&ID diagrams, and maintenance logs. Answering a question like *"If pump CW-P-101 fails, what's affected downstream?"* requires understanding equipment relationships, not just matching keywords in a document.

FixMyPlant solves this with **three retrieval routes**, automatically selected (or fused) per query:

| Route | Strategy | Best For | Example |
|-------|----------|----------|---------|
| **Vector** | Semantic similarity search over document chunks | Spec/requirement lookups | *"What are the water quality requirements for BFW?"* |
| **Graph Local** | Equipment topology traversal (BFS over `CONNECTS_TO`) | Impact analysis | *"If CW-P-101 fails, what's affected downstream?"* |
| **Graph Global** | Community summary aggregation (Louvain-style clustering) | Cross-system patterns | *"Common causes of shutdowns across all systems?"* |
| **Hybrid** | Weighted Reciprocal Rank Fusion of Vector + Graph Local | General-purpose queries | *"Explain the startup sequence for STG-101"* |

Every answer includes **inline citations** back to the source document and chunk, so operators can verify claims against the original manual.

## Architecture

FixMyPlant follows a **hexagonal (ports & adapters)** architecture. The domain layer has zero third-party dependencies; all infrastructure concerns (LLMs, databases, HTTP) are isolated behind Protocol interfaces.

```
                        ┌─────────────────────┐
                        │   Streamlit UI       │  inbound adapter
                        │  (chat, graph viz)    │
                        └──────────┬───────────┘
                                   │ HTTP
                        ┌──────────▼───────────┐
                        │   FastAPI Adapter     │  inbound adapter
                        │  /query /ingest /health│
                        └──────────┬───────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │        Domain (pure logic)          │
                 │  entities · HybridContextFuser       │
                 └─────────────────┼─────────────────┘
                                   │  ports (Protocols)
        ┌──────────────┬───────────┼────────────┬──────────────┐
        ▼              ▼           ▼            ▼              ▼
  Neo4j Storage   Ollama/LangGraph  Vision OCR  Composite    In-Memory
  Adapter         Orchestrator      (Gemini)    Parser       Storage
  (vector+graph)  (route, embed,               (fallback for  (fallback if
                   synthesize)                  .pdf/.png)     Neo4j is down)
```

- **`src/domain/`** — Frozen dataclass entities (`DocumentChunk`, `EquipmentNode`, `SynthesizedResponse`, ...) and `HybridContextFuser`. No imports outside the standard library.
- **`src/ports/`** — `Protocol` interfaces splitting inbound (`KnowledgeQueryPort`, `DocumentIngestionPort`) from outbound (`VectorStoragePort`, `GraphStoragePort`, `LLMInferencePort`, `DocumentParsingPort`) dependencies.
- **`src/adapters/inbound/`** — FastAPI HTTP adapter, Streamlit chat UI, batch file uploader.
- **`src/adapters/outbound/`** — Neo4j (unified vector + graph store), LangGraph/Ollama orchestrator, Gemini Vision OCR parser, text parser, in-memory fallback store.
- **`src/main.py`** — Composition root. Wires adapters together and falls back gracefully (e.g., in-memory storage if Neo4j is unreachable, text-only parsing if `GEMINI_API_KEY` is unset).

This means you can swap Neo4j for another graph database, or Ollama for a hosted LLM, by writing a new adapter — the domain and API layer never change.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph + vector storage | [Neo4j 5](https://neo4j.com/) (native vector index) |
| LLM inference & embeddings | [Ollama](https://ollama.ai/) — `qwen2.5:7b-instruct` + `nomic-embed-text` |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Document OCR (optional) | [Google Gemini](https://ai.google.dev/) multimodal extraction |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Frontend | [Streamlit](https://streamlit.io/) + [pyvis](https://pyvis.readthedocs.io/) graph visualization |
| Testing | [pytest](https://docs.pytest.org/) + [Hypothesis](https://hypothesis.readthedocs.io/) (property-based tests) |
| Deployment | Docker + Docker Compose |

## Getting Started

### Option A — Docker (recommended)

Everything — Neo4j, Ollama, the API, and the UI — runs in containers. No local Python or database installation required.

**Prerequisites**: [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

**1. Configure environment**

```bash
git clone <repo-url> fixmyplant
cd fixmyplant
cp .env.example .env
# Optional: set GEMINI_API_KEY in .env to enable PDF/image ingestion via Vision OCR
```

**2. Build and start the stack**

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

| Service | Container | Host Port | Role |
|---------|-----------|-----------|------|
| `neo4j` | `fixmyplant-neo4j` | `7474` (browser), `7687` (bolt) | Graph + vector storage |
| `ollama` | `fixmyplant-ollama` | *(internal only)* | Local LLM inference |
| `api` | `fixmyplant-api` | `8000` | FastAPI backend |
| `streamlit` | `fixmyplant-streamlit` | `8501` | Chat UI |

Wait for all containers to report `healthy`:

```bash
docker compose -f infra/docker-compose.yml ps
```

**3. Pull the Ollama models** (first run only — persisted in the `ollama_data` volume afterwards)

```bash
docker exec fixmyplant-ollama ollama pull qwen2.5:7b-instruct
docker exec fixmyplant-ollama ollama pull nomic-embed-text
```

**4. Initialize the database and load the sample dataset**

```bash
docker exec fixmyplant-api python scripts/init_neo4j_schema.py
docker exec fixmyplant-api python scripts/ingest_golden_dataset.py
docker exec fixmyplant-api python scripts/rebuild_communities.py
```

> `rebuild_communities.py` calls the local LLM to summarize equipment clusters. On CPU-only hosts this can take a few minutes — this is expected.

**5. Open the app**

| | |
|---|---|
| 💬 Chat UI | http://localhost:8501 |
| 📚 API docs (Swagger) | http://localhost:8000/docs |
| 🗄️ Neo4j Browser | http://localhost:7474 (user `neo4j`, password from `.env`) |

### Option B — Local Python

For active development without rebuilding Docker images on every change.

**Prerequisites**

- Docker (for Neo4j only — via `infra/docker-compose.local.yml`)
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally
- Poppler (for PDF rasterization): `sudo apt install poppler-utils` (Linux) · `brew install poppler` (macOS)
- A Gemini API key (optional, for PDF/image ingestion)

**Setup**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your GEMINI_API_KEY and any custom credentials

docker compose -f infra/docker-compose.local.yml up -d   # Neo4j only
python scripts/init_neo4j_schema.py

ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

python scripts/ingest_golden_dataset.py
python scripts/rebuild_communities.py

uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
streamlit run src/adapters/inbound/streamlit_adapter.py
```

## Configuration

All configuration is environment-driven (see `src/config.py`). Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `local` | `local`, `test`, or `production` |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection string (`bolt://neo4j:7687` inside Docker) |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | **Required.** Neo4j password |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint (`http://ollama:11434` inside Docker) |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Chat/generation model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `GEMINI_API_KEY` | — | Optional. Enables Vision OCR for PDF/image ingestion |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for OCR extraction |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | FastAPI bind address |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL the Streamlit UI calls |

Without `GEMINI_API_KEY`, the app still works fully for `.txt`/`.md`/`.csv`/`.log` ingestion — only PDF and image OCR is disabled.

## Usage

Once running, ask questions in the chat UI or via the API directly:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the trip conditions for the steam turbine?"}'
```

Try the built-in demo questions to exercise each retrieval route:

1. **Vector search** — *"What are the trip conditions for the steam turbine?"*
2. **Graph local** — *"If pump CW-P-101 fails, what downstream equipment is affected?"*
3. **Graph global** — *"What are the most common causes of unplanned shutdowns across all systems?"*
4. **Hybrid** — *"Explain the startup sequence for STG-101 and its dependencies"*

Upload your own documents (`.txt`, `.md`, `.pdf`, `.png`, `.jpg`, `.tiff`, `.bmp`) via the sidebar in the Streamlit UI, or `POST /ingest`.

## API Reference

Full interactive docs are available at `/docs` (Swagger UI) once the API is running. Summary:

| Method | Endpoint | Description |
|--------|----------|--------------|
| `POST` | `/query` | Ask a question. Body: `{"question": str, "history"?: list[dict]}`. Returns `{answer, route_used, citations}` |
| `POST` | `/ingest` | Upload a document (multipart `file`). Parses, embeds, and stores it |
| `GET` | `/health` | Liveness + dependency status: `{status, neo4j, ollama}` |
| `GET` | `/graph` | Returns `{nodes, edges}` for the equipment knowledge graph visualization |

> ⚠️ **Security note**: These endpoints have no authentication or rate limiting by design (hackathon/demo scope). If deploying beyond local/trusted networks, add an auth layer (e.g., API keys, OAuth2) and a reverse proxy before exposing `/ingest` or `/query` publicly.

## Project Structure

```
├── src/
│   ├── domain/              # Entities, services — pure logic, zero third-party deps
│   ├── ports/                # Protocol interfaces (inbound + outbound)
│   ├── adapters/
│   │   ├── inbound/           # FastAPI, Streamlit, batch uploader
│   │   └── outbound/          # Neo4j, Ollama/LangGraph, Gemini OCR, text parser
│   ├── config.py              # Environment-driven settings
│   └── main.py                 # Composition root (wires everything together)
├── scripts/                  # One-time setup and data ingestion scripts
├── data/golden_dataset/      # Sample industrial documents (P&ID logs, manuals)
├── infra/
│   ├── docker-compose.yml         # Full stack: neo4j + ollama + api + streamlit
│   ├── docker-compose.local.yml   # Neo4j only, for local Python development
│   ├── Dockerfile.api             # FastAPI backend image
│   └── Dockerfile.streamlit       # Streamlit UI image
├── tests/
│   ├── unit/                  # Domain and adapter unit tests, property-based tests
│   └── integration/            # End-to-end tests requiring a live Neo4j instance
└── .streamlit/                # Streamlit theme configuration
```

## Operating the Docker Stack

```bash
# Tail logs for a specific service
docker compose -f infra/docker-compose.yml logs -f api
docker compose -f infra/docker-compose.yml logs -f streamlit

# Stop containers (named volumes — and their data — persist)
docker compose -f infra/docker-compose.yml down

# Stop and wipe all data (Neo4j graph + Ollama models)
docker compose -f infra/docker-compose.yml down -v

# Rebuild images after a code change
docker compose -f infra/docker-compose.yml up -d --build
```

**Notes**

- Port `11434` (Ollama) is intentionally **not** published to the host, to avoid clashing with a locally-installed Ollama service. The `api` container reaches it over the internal Docker network at `http://ollama:11434`.
- For GPU acceleration, add a `deploy.resources.reservations.devices` block to the `ollama` service — see the [Ollama Docker image docs](https://hub.docker.com/r/ollama/ollama).
- Ollama and Neo4j both persist to named volumes (`ollama_data`, `neo4j_data`, `neo4j_logs`), so container restarts don't require re-pulling models or re-ingesting data.

## Testing

```bash
ENVIRONMENT=test pytest tests/ -v
```

- **Unit tests** cover domain logic, hybrid fusion ranking, and adapter behavior in isolation.
- **Property-based tests** (Hypothesis) validate batch ingestion sequencing invariants.
- **Integration tests** exercise the real Neo4j adapter and are automatically skipped if no database is reachable — no manual flag needed.

To run integration tests locally, start Neo4j first:

```bash
docker compose -f infra/docker-compose.local.yml up -d
ENVIRONMENT=test pytest tests/integration/ -v
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `/query` returns 500 with `embed failed ... 404` | Ollama models not pulled yet | Run the `ollama pull` commands in [step 3](#option-a--docker-recommended) |
| `/query` returns 500 with `no such vector schema index` | Neo4j schema not initialized | Run `scripts/init_neo4j_schema.py` |
| `/health` shows `"neo4j": false` | Neo4j container not healthy / wrong credentials | Check `docker compose ps` and that `.env` matches `NEO4J_AUTH` |
| Answers are empty or generic | No data ingested yet | Run `scripts/ingest_golden_dataset.py` or upload documents via the UI |
| `rebuild_communities.py` is slow | Local LLM inference on CPU | Expected — allow a few minutes, or use a GPU-enabled Ollama container |
| PDF/image uploads silently skipped | `GEMINI_API_KEY` not set | Add a Gemini API key to `.env` and restart the `api` container |
| Docker port conflict on `11434` | A local Ollama service already binds that port | Already handled — the container doesn't publish `11434` to the host |

## Roadmap

- [ ] Authentication and rate limiting for public deployments
- [ ] Streaming responses over WebSocket/SSE
- [ ] Multi-tenant document namespaces
- [ ] Automated community rebuild on ingestion (currently manual)
- [ ] CI pipeline (lint, type-check, test) on pull requests

## Contributing

Contributions are welcome. Please:

1. Fork the repo and create a feature branch (`git checkout -b feature/my-change`)
2. Keep the hexagonal boundaries intact — domain code must not import adapters or vendor SDKs
3. Add or update tests for any behavior change (`pytest tests/ -v`)
4. Open a pull request with a clear description of the change and testing performed

## License

MIT — see [LICENSE](LICENSE) for details.

Originally built as a hackathon project (ET AI Industrial Brain).
