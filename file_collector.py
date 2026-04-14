"""
File Collector
==============
Pre-scan repository source files and assign smell injection targets
based on file size (line count).

Each eligible file produces exactly 1 case per smell type.
With 6 smell types × 3 difficulty levels the combinatorial space is
already large, so we keep file selection strict and avoid duplicates.
"""

import os
from typing import Dict, List, Set

# Skip these directory names during traversal
SKIP_DIRS = {
    "__pycache__", ".git", ".tox", ".mypy_cache", ".pytest_cache",
    "node_modules", "venv", ".venv", "env", ".env",
    "site-packages", "dist-packages", "egg-info",
}

# Skip files in directories matching these substrings
SKIP_DIR_SUBSTRINGS = {"test", "tests", "testing", "conftest"}

# Minimum line count for a file to be eligible — focus on substantial core files
MIN_LINES = 500

# Skip these file names — utility / glue / compat files are too fragmented
# for meaningful smell injection
SKIP_FILE_NAMES = {
    "utils.py", "util.py", "utilities.py",
    "helpers.py", "helper.py",
    "compat.py", "compat_utils.py",
    "constants.py", "consts.py",
    "exceptions.py", "errors.py",
    "types.py", "typing.py", "_typing.py",
    "conftest.py",
    "setup.py", "conf.py", "config.py",
    "version.py", "_version.py",
    "deprecations.py", "warnings.py",
}


def _count_lines(file_path: str) -> int:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def collect_source_files(repo_path: str, src_path: str) -> List[Dict]:
    """Scan the repo's source directory and collect eligible Python files.

    Only files with >= MIN_LINES lines are included. Each file produces
    exactly 1 case per smell type (no multi-case based on size).

    Args:
        repo_path: Absolute path to the repository root.
        src_path: Relative path to source code within the repo (e.g. "src/click").

    Returns:
        List of dicts sorted by line count descending:
        [{"file": relative_path, "abs_path": abs_path, "lines": N}, ...]
    """
    scan_root = os.path.join(repo_path, src_path)
    if not os.path.isdir(scan_root):
        return []

    results = []

    for dirpath, dirnames, filenames in os.walk(scan_root):
        # Filter out directories we should skip
        rel_dir = os.path.relpath(dirpath, repo_path)
        dir_parts = set(rel_dir.split(os.sep))

        # Skip hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        # Skip known non-source directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        # Skip test directories
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIR_SUBSTRINGS
        ]

        # Check if current directory itself is a test directory
        if dir_parts & SKIP_DIR_SUBSTRINGS:
            continue

        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname == "__init__.py":
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            if fname.lower() in SKIP_FILE_NAMES:
                continue
            if fname.startswith("_") and fname != "__init__.py":
                continue

            abs_path = os.path.join(dirpath, fname)
            lines = _count_lines(abs_path)

            if lines < MIN_LINES:
                continue

            rel_path = os.path.relpath(abs_path, repo_path)

            results.append({
                "file": rel_path,
                "abs_path": abs_path,
                "lines": lines,
            })

    # Sort by line count descending (largest files first)
    results.sort(key=lambda x: x["lines"], reverse=True)
    return results


def assign_targets(
    files: List[Dict],
    smell_types: List[Dict],
    completed_assignments: Set[str] | None = None,
) -> List[Dict]:
    """Assign (file, smell_type) pairs for generation.

    Each smell type gets exactly 1 case per eligible file.

    Args:
        files: Output of collect_source_files().
        smell_types: List of smell type dicts (must have "type" key).
        completed_assignments: Set of assignment keys to skip
            (format: "{smell_type}::{file}").

    Returns:
        List of assignment dicts:
        [{"file": path, "abs_path": abs, "lines": N, "smell_type": type,
          "smell_desc": desc, "key": unique_key}, ...]
    """
    completed = completed_assignments or set()
    assignments = []

    for smell in smell_types:
        smell_type = smell["type"]
        smell_desc = smell.get("desc", "")

        for finfo in files:
            key = f"{smell_type}::{finfo['file']}"
            if key in completed:
                continue

            assignments.append({
                "file": finfo["file"],
                "abs_path": finfo["abs_path"],
                "lines": finfo["lines"],
                "smell_type": smell_type,
                "smell_desc": smell_desc,
                "smell_config": smell,
                "key": key,
            })

    return assignments


def build_completed_keys(refactor_codes: List[Dict]) -> Set[str]:
    """Extract assignment keys from previously completed results.

    Matches on smell type + target file + case_index stored in each result.
    """
    keys = set()
    for entry in refactor_codes:
        key = entry.get("assignment_key")
        if key:
            keys.add(key)
    return keys
