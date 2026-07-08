"""Domain purity validation test.

Scans all .py files in src/domain/ and src/ports/ directories and asserts
zero imports from any third-party package. Only Python standard library
and intra-project imports are permitted.

Validates: Requirements 9.1, 9.2, 9.3, 9.4
"""

import ast
import sys
from pathlib import Path

# Known Python standard library top-level module names (Python 3.10+).
# This is a comprehensive but not exhaustive list — covers the most common
# modules. We use sys.stdlib_module_names on Python 3.10+ for accuracy.
if hasattr(sys, "stdlib_module_names"):
    STDLIB_MODULES: set[str] = set(sys.stdlib_module_names)
else:
    # Fallback for Python <3.10 — manually maintained set
    STDLIB_MODULES = {
        "__future__", "abc", "aifc", "argparse", "array", "ast", "asynchat",
        "asyncio", "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
        "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
        "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
        "colorsys", "compileall", "concurrent", "configparser", "contextlib",
        "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv", "ctypes",
        "curses", "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
        "distutils", "doctest", "email", "encodings", "enum", "errno",
        "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch", "fractions",
        "ftplib", "functools", "gc", "getopt", "getpass", "gettext", "glob",
        "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http", "idlelib",
        "imaplib", "imghdr", "imp", "importlib", "inspect", "io", "ipaddress",
        "itertools", "json", "keyword", "lib2to3", "linecache", "locale",
        "logging", "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes",
        "mmap", "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
        "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
        "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
        "plistlib", "poplib", "posix", "posixpath", "pprint", "profile",
        "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue",
        "quopri", "random", "re", "readline", "reprlib", "resource", "rlcompleter",
        "runpy", "sched", "secrets", "select", "selectors", "shelve", "shlex",
        "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr", "socket",
        "socketserver", "spwd", "sqlite3", "sre_compile", "sre_constants",
        "sre_parse", "ssl", "stat", "statistics", "string", "stringprep",
        "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig",
        "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile", "termios",
        "test", "textwrap", "threading", "time", "timeit", "tkinter", "token",
        "tokenize", "tomllib", "trace", "traceback", "tracemalloc", "tty",
        "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest",
        "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
        "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml",
        "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib", "_thread",
        "typing_extensions",
    }

# Packages explicitly banned by the spec
BANNED_PACKAGES: set[str] = {
    "neo4j", "langgraph", "fastapi", "streamlit", "ollama",
}

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _get_all_py_files(*dirs: str) -> list[Path]:
    """Collect all .py files from the given directories relative to project root."""
    files: list[Path] = []
    for d in dirs:
        dir_path = PROJECT_ROOT / d
        if dir_path.exists():
            files.extend(dir_path.rglob("*.py"))
    return files


def _extract_imports(filepath: Path) -> list[str]:
    """Parse a Python file and extract all top-level imported module names.

    For `import X.Y.Z`, extracts 'X'.
    For `from X.Y import Z`, extracts 'X'.
    Handles TYPE_CHECKING guarded imports as well by parsing the full AST.
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))

    top_level_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import X.Y.Z` → top-level is 'X'
                top_module = alias.name.split(".")[0]
                top_level_modules.append(top_module)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                # `from X.Y import Z` → top-level is 'X'
                top_module = node.module.split(".")[0]
                top_level_modules.append(top_module)
            # `from . import X` (relative) — these are intra-project, skip

    return top_level_modules


def _is_allowed_import(module_name: str) -> bool:
    """Check if a top-level module name is allowed (stdlib or intra-project)."""
    # Intra-project imports: 'src' package
    if module_name == "src":
        return True
    # Standard library
    if module_name in STDLIB_MODULES:
        return True
    return False


class TestDomainPurity:
    """Validate that src/domain/ and src/ports/ contain no third-party imports."""

    def test_no_banned_package_imports(self) -> None:
        """Assert zero imports from explicitly banned packages."""
        py_files = _get_all_py_files("src/domain", "src/ports")
        assert py_files, "No .py files found in src/domain/ or src/ports/"

        violations: list[str] = []
        for filepath in py_files:
            imports = _extract_imports(filepath)
            for mod in imports:
                if mod in BANNED_PACKAGES:
                    rel_path = filepath.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel_path}: imports banned package '{mod}'")

        assert not violations, (
            "Banned third-party imports found in domain/ports layer:\n"
            + "\n".join(violations)
        )

    def test_only_stdlib_and_intraproject_imports(self) -> None:
        """Assert all imports are either standard library or intra-project (src.)."""
        py_files = _get_all_py_files("src/domain", "src/ports")
        assert py_files, "No .py files found in src/domain/ or src/ports/"

        violations: list[str] = []
        for filepath in py_files:
            imports = _extract_imports(filepath)
            for mod in imports:
                if not _is_allowed_import(mod):
                    rel_path = filepath.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"{rel_path}: imports non-stdlib package '{mod}'"
                    )

        assert not violations, (
            "Third-party imports found in domain/ports layer:\n"
            + "\n".join(violations)
        )

    def test_type_checking_guarded_imports_also_clean(self) -> None:
        """Assert that TYPE_CHECKING guarded imports also have no vendor refs.

        The AST walker in _extract_imports already walks into if-blocks
        including TYPE_CHECKING guards, so this test validates the same
        constraint applies to those guarded imports (Requirement 9.4).
        """
        py_files = _get_all_py_files("src/domain", "src/ports")
        assert py_files, "No .py files found in src/domain/ or src/ports/"

        violations: list[str] = []
        for filepath in py_files:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(filepath))

            # Find if TYPE_CHECKING blocks
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # Check if the test is `TYPE_CHECKING` or `typing.TYPE_CHECKING`
                    is_type_checking = False
                    if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                        is_type_checking = True
                    elif (
                        isinstance(node.test, ast.Attribute)
                        and isinstance(node.test.value, ast.Name)
                        and node.test.value.id == "typing"
                        and node.test.attr == "TYPE_CHECKING"
                    ):
                        is_type_checking = True

                    if is_type_checking:
                        # Walk inside the if block for imports
                        for child in ast.walk(node):
                            if isinstance(child, ast.Import):
                                for alias in child.names:
                                    top_mod = alias.name.split(".")[0]
                                    if not _is_allowed_import(top_mod):
                                        rel_path = filepath.relative_to(PROJECT_ROOT)
                                        violations.append(
                                            f"{rel_path}: TYPE_CHECKING imports "
                                            f"non-stdlib package '{top_mod}'"
                                        )
                            elif isinstance(child, ast.ImportFrom):
                                if child.module is not None:
                                    top_mod = child.module.split(".")[0]
                                    if not _is_allowed_import(top_mod):
                                        rel_path = filepath.relative_to(PROJECT_ROOT)
                                        violations.append(
                                            f"{rel_path}: TYPE_CHECKING imports "
                                            f"non-stdlib package '{top_mod}'"
                                        )

        assert not violations, (
            "Banned imports in TYPE_CHECKING blocks:\n" + "\n".join(violations)
        )

    def test_scanned_directories_contain_python_files(self) -> None:
        """Sanity check: verify we actually found files to scan."""
        domain_files = _get_all_py_files("src/domain")
        ports_files = _get_all_py_files("src/ports")

        assert len(domain_files) >= 1, "Expected at least 1 .py file in src/domain/"
        assert len(ports_files) >= 1, "Expected at least 1 .py file in src/ports/"
