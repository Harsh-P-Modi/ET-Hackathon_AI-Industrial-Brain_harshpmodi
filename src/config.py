"""
Environment-driven configuration for FixMyPlant.

Reads settings from environment variables (optionally loaded from a .env file).
Uses only the Python standard library — no third-party config libraries.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_DIMENSIONS: int = 768  # Locked to nomic-embed-text model


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_dotenv_if_exists(dotenv_path: Path | None = None) -> None:
    """
    Read a .env file (simple key=value format) and inject entries into
    os.environ. Skips blank lines and comments (lines starting with #).

    If *dotenv_path* is None, defaults to `.env` in the project root.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"

    if not dotenv_path.is_file():
        return

    with open(dotenv_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip optional surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Load .env on module import (best-effort)
# ---------------------------------------------------------------------------

load_dotenv_if_exists()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "local")

NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.environ.get("NEO4J_USER", "neo4j")

_neo4j_password = os.environ.get("NEO4J_PASSWORD")
if _neo4j_password is None and ENVIRONMENT != "test":
    raise RuntimeError(
        "NEO4J_PASSWORD environment variable is required. "
        "Set it in your .env file or export it before running the application."
    )
NEO4J_PASSWORD: str = _neo4j_password or ""

# ---------------------------------------------------------------------------
# Ollama LLM Settings
# ---------------------------------------------------------------------------

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_EMBED_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Gemini Vision OCR Settings
# ---------------------------------------------------------------------------

_gemini_api_key = os.environ.get("GEMINI_API_KEY")
if _gemini_api_key is None and ENVIRONMENT != "test":
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "GEMINI_API_KEY not set — Vision OCR will be disabled. "
        "Only text file ingestion will work. Set it in .env if you need PDF/image parsing."
    )
GEMINI_API_KEY: str = _gemini_api_key or ""

GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# ---------------------------------------------------------------------------
# FastAPI Settings
# ---------------------------------------------------------------------------

API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("API_PORT", "8000"))
