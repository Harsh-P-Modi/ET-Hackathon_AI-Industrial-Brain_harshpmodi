"""Walk data/golden_dataset/ and ingest every file via BatchFileUploaderAdapter.

Usage:
    python scripts/ingest_golden_dataset.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from neo4j import GraphDatabase  # noqa: E402

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  # noqa: E402
from src.adapters.outbound.neo4j_storage_adapter import Neo4jUnifiedStorageAdapter  # noqa: E402
from src.adapters.outbound.ollama_llm_adapter import LangGraphOrchestratorAdapter  # noqa: E402
from src.adapters.outbound.vision_ocr_adapter import VisionOCRAdapter  # noqa: E402
from src.adapters.inbound.batch_uploader_adapter import BatchFileUploaderAdapter  # noqa: E402
from src.domain.services import HybridContextFuser  # noqa: E402

GOLDEN_DATASET_DIR = _project_root / "data" / "golden_dataset"


def main() -> None:
    if not GOLDEN_DATASET_DIR.is_dir():
        print(f"ERROR: Golden dataset directory not found: {GOLDEN_DATASET_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Neo4j at {NEO4J_URI} as '{NEO4J_USER}' ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    storage = Neo4jUnifiedStorageAdapter(driver)
    fuser = HybridContextFuser()
    llm = LangGraphOrchestratorAdapter(vector_store=storage, graph_store=storage, fuser=fuser)
    parser = VisionOCRAdapter()
    adapter = BatchFileUploaderAdapter(parser=parser, llm=llm, storage=storage)

    failures: list[tuple[str, str]] = []

    files = sorted(GOLDEN_DATASET_DIR.iterdir())
    print(f"Found {len(files)} files in {GOLDEN_DATASET_DIR}")

    for filepath in files:
        if filepath.is_file():
            print(f"Ingesting: {filepath.name} ...")
            try:
                raw_bytes = filepath.read_bytes()
                adapter.ingest(raw_bytes, filepath.name)
                print(f"  -> OK")
            except Exception as exc:
                failures.append((filepath.name, str(exc)))
                print(f"  -> FAILED: {exc}")

    if failures:
        print(f"\n{len(failures)} file(s) failed:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        driver.close()
        sys.exit(1)
    else:
        print(f"\nAll files ingested successfully.")

    driver.close()


if __name__ == "__main__":
    main()
