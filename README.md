# 🏭 FixMyPlant — AI Industrial Brain

A Retrieval-Augmented Generation (RAG) system for industrial plant maintenance. Ask natural-language questions about your equipment and get grounded answers backed by P&ID diagrams, maintenance logs, and safety manuals.

## Architecture

Three retrieval routes fused into a single answer:

| Route | Strategy | Example Question |
|-------|----------|-----------------|
| A — Vector | Semantic similarity search | "What are the water quality requirements for BFW?" |
| B — Graph Local | Equipment topology traversal | "If CW-P-101 fails, what's affected downstream?" |
| C — Graph Global | Community summary aggregation | "Common causes of shutdowns across all systems?" |
| Hybrid | Weighted Reciprocal Rank Fusion | General questions → combines A + B |

**Stack**: Neo4j 5 · Ollama (qwen2.5 + nomic-embed-text) · Google Gemini (OCR) · FastAPI · Streamlit · LangGraph

The app is fully containerized: Neo4j, Ollama, the FastAPI backend, and the Streamlit UI each run in their own Docker container, orchestrated with Docker Compose.

## Prerequisites

- **Docker** and **Docker Compose** ([install guide](https://docs.docker.com/get-docker/))
- **Gemini API key** (optional — only needed to ingest PDF/image documents via Vision OCR; `.txt` ingestion works without it)

That's it. Python, Neo4j, and Ollama all run inside containers — no local installation required.

## Quick Start (Docker)

### 1. Clone and configure environment

```bash
cd ET-Hackathon_AI-Industrial-Brain_harshpmodi
cp .env.example .env
# Edit .env — fill in GEMINI_API_KEY if you plan to ingest PDFs/images
```

### 2. Build and start all services

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

This starts four containers:

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| `neo4j` | `fixmyplant-neo4j` | 7474 (browser), 7687 (bolt) | Graph + vector storage |
| `ollama` | `fixmyplant-ollama` | internal only | Local LLM inference |
| `api` | `fixmyplant-api` | 8000 | FastAPI backend |
| `streamlit` | `fixmyplant-streamlit` | 8501 | Chat UI |

Check status and wait for all containers to become healthy:

```bash
docker compose -f infra/docker-compose.yml ps
```

### 3. Pull the Ollama models into the container

The `ollama` container starts empty — models are pulled once and persisted in a named volume (`ollama_data`), so this is only needed the first time:

```bash
docker exec fixmyplant-ollama ollama pull qwen2.5:7b-instruct
docker exec fixmyplant-ollama ollama pull nomic-embed-text
```

### 4. Initialize the Neo4j schema

```bash
docker exec fixmyplant-api python scripts/init_neo4j_schema.py
```

### 5. Ingest the golden dataset

```bash
docker exec fixmyplant-api python scripts/ingest_golden_dataset.py
```

This parses all documents in `data/golden_dataset/` (Gemini Vision for PDFs/images, local parsing for `.txt`) and stores structured knowledge in Neo4j.

### 6. Build community summaries

```bash
docker exec fixmyplant-api python scripts/rebuild_communities.py
```

This step calls the local LLM to summarize equipment clusters — it can take a few minutes on CPU.

### 7. Open the app

- **Chat UI**: http://localhost:8501
- **API docs**: http://localhost:8000/docs
- **Neo4j browser**: http://localhost:7474 (user: `neo4j`, password: from `.env`)

## Managing the Stack

```bash
# View logs
docker compose -f infra/docker-compose.yml logs -f api
docker compose -f infra/docker-compose.yml logs -f streamlit

# Stop containers (data persists in volumes)
docker compose -f infra/docker-compose.yml down

# Stop and wipe all data (Neo4j graph + Ollama models)
docker compose -f infra/docker-compose.yml down -v

# Rebuild after code changes
docker compose -f infra/docker-compose.yml up -d --build
```

### Notes on Ollama and ports

- Port `11434` is **not** published to the host, to avoid conflicting with a locally-installed Ollama service. The `api` container reaches Ollama internally via `http://ollama:11434`.
- If you have GPU acceleration available, add a `deploy.resources.reservations.devices` block to the `ollama` service in `infra/docker-compose.yml` (see the [Ollama Docker docs](https://hub.docker.com/r/ollama/ollama)).

## Running Without Docker (local Python)

If you prefer to run the app directly on your machine:

### Prerequisites

- **Docker** (for Neo4j only)
- **Python 3.11+**
- **Ollama** installed and running locally ([install guide](https://ollama.ai))
- **Poppler** (for PDF rasterization): `sudo apt install poppler-utils` (Linux) or `brew install poppler` (macOS)
- **Gemini API key** (for document ingestion only)

### Steps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — fill in your GEMINI_API_KEY and adjust passwords if needed

docker compose -f infra/docker-compose.local.yml up -d
python scripts/init_neo4j_schema.py

ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

python scripts/ingest_golden_dataset.py
python scripts/rebuild_communities.py

uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

In a separate terminal:

```bash
streamlit run src/adapters/inbound/streamlit_adapter.py
```

Open http://localhost:8501 — the chat UI connects to the FastAPI backend.

## Project Structure

```
├── src/
│   ├── domain/          # Entities, services (pure logic, no deps)
│   ├── ports/           # Protocol interfaces (inbound + outbound)
│   ├── adapters/
│   │   ├── inbound/     # FastAPI, Streamlit, Batch uploader
│   │   └── outbound/    # Neo4j, Ollama/LangGraph, Gemini OCR
│   ├── config.py        # Environment-driven settings
│   └── main.py          # Composition root (wires everything)
├── scripts/             # One-time setup and data ingestion
├── data/golden_dataset/ # Sample industrial documents
├── infra/
│   ├── docker-compose.yml        # Full stack: neo4j + ollama + api + streamlit
│   ├── docker-compose.local.yml  # Neo4j only, for local Python development
│   ├── Dockerfile.api            # FastAPI backend image
│   └── Dockerfile.streamlit      # Streamlit UI image
├── tests/               # Unit + integration + property-based tests
└── .streamlit/           # Theme configuration
```

## Running Tests

```bash
ENVIRONMENT=test pytest tests/ -v
```

The 6 integration tests that require a live Neo4j instance will be skipped if the database is unreachable.

## Demo Questions to Try

1. **Vector search**: "What are the trip conditions for the steam turbine?"
2. **Graph local**: "If pump CW-P-101 fails, what downstream equipment is affected?"
3. **Graph global**: "What are the most common causes of unplanned shutdowns across all systems?"
4. **Hybrid**: "Explain the startup sequence for STG-101 and its dependencies"

## License

Hackathon project — ET AI Industrial Brain.
