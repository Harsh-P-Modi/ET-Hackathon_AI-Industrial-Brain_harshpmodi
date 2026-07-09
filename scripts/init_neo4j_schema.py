"""
One-time idempotent Neo4j schema migration script.

Creates vector indexes and uniqueness constraints required by FixMyPlant.
Safe to re-run — all statements use IF NOT EXISTS.

Usage:
    python scripts/init_neo4j_schema.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from neo4j import GraphDatabase  # noqa: E402

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  # noqa: E402


def main() -> None:
    print(f"Connecting to Neo4j at {NEO4J_URI} as '{NEO4J_USER}' ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("Connection verified.\n")

        with driver.session() as session:
            # 1. Vector index on Chunk.embedding
            print("Creating vector index 'chunk_embeddings' (IF NOT EXISTS) ...")
            session.run(
                """
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: 768,
                  `vector.similarity_function`: 'cosine'
                }}
                """
            )
            print("  -> Done.\n")

            # 2. Uniqueness constraint on Equipment.equipment_id
            print("Creating constraint 'equipment_id_unique' (IF NOT EXISTS) ...")
            session.run(
                """
                CREATE CONSTRAINT equipment_id_unique IF NOT EXISTS
                FOR (e:Equipment) REQUIRE e.equipment_id IS UNIQUE
                """
            )
            print("  -> Done.\n")

            # 3. Uniqueness constraint on Chunk.chunk_id
            print("Creating constraint 'chunk_id_unique' (IF NOT EXISTS) ...")
            session.run(
                """
                CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
                """
            )
            print("  -> Done.\n")

        print("Schema migration complete.")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        driver.close()
        print("Driver closed.")


if __name__ == "__main__":
    main()
