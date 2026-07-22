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

## Prerequisites

- **Docker** (for Neo4j)
- **Python 3.11+**
- **Ollama** installed and running locally ([install guide](https://ollama.ai))
- **Poppler** (for PDF rasterization): `brew install poppler`
- **Gemini API key** (for document ingestion only)

## Quick Start

### 1. Clone and install

```bash
cd ET-Hackathon_AI-Industrial-Brain_harshpmodi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment setup

```bash
cp .env.example .env
# Edit .env — fill in your GEMINI_API_KEY and adjust passwords if needed
```

### 3. Start Neo4j

```bash
docker compose -f infra/docker-compose.local.yml up -d
```

Wait ~15 seconds for Neo4j to become healthy, then initialize the schema:

```bash
python scripts/init_neo4j_schema.py
```

### 4. Pull Ollama models

```bash
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

### 5. Ingest the golden dataset

```bash
python scripts/ingest_golden_dataset.py
```

This parses all documents in `data/golden_dataset/` via Gemini Vision, embeds them, and stores structured knowledge in Neo4j.

### 6. Build community summaries

```bash
python scripts/rebuild_communities.py
```

### 7. Run the backend

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 8. Run the Streamlit UI

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
├── infra/               # Docker Compose for Neo4j
├── tests/               # Unit + integration + property-based tests
└── .streamlit/          # Theme configuration
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
