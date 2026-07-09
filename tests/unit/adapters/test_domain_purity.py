"""Domain purity test for vendor import ban.

Scans all .py files in src/domain/ and src/ports/ directories and asserts
that NONE of them contain import statements for banned packages:
langgraph, ollama, requests, neo4j.

Validates: Requirements 6.1, 6.3
"""

import re
from pathlib import Path

import pytest

# Banned imports that must never appear in domain or ports layers.
BANNED_PACKAGES = ("langgraph", "ollama", "requests", "neo4j")

# Pattern matches: "import neo4j", "from neo4j import ...", "import requests", etc.
IMPORT_PATTERN = re.compile(
    r"^\s*(import|from)\s+(" + "|".join(BANNED_PACKAGES) + r")\b",
    re.MULTILINE,
)


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]  # tests/unit/adapters/ -> project root


def test_no_vendor_imports_in_domain_ports():
    """Assert zero imports from banned vendor packages in domain/ports layer.

    Banned packages: langgraph, ollama, requests, neo4j.
    These must only appear in src/adapters/, never in domain or ports.
    """
    root = _get_project_root()
    violations = []

    for directory in ("src/domain", "src/ports"):
        dir_path = root / directory
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            matches = IMPORT_PATTERN.findall(content)
            for match in matches:
                violations.append(
                    f"{py_file.relative_to(root)}: {match[0]} {match[1]}"
                )

    assert not violations, (
        f"Vendor imports found in domain/ports layer:\n" + "\n".join(violations)
    )
