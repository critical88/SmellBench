#!/usr/bin/env python
"""
Build a method-to-test tree for click by running coverage with per-test contexts.

This script discovers and runs the pytest suite that lives in ``../project/click`` (relative
to this file), records which implementation methods are exercised by each test, and prints
an ASCII tree whose leaves are the test cases. Give it a method key and it will list all
tests connected to that method.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set
import subprocess

try:
    import coverage  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "The coverage package is required. Install it with `pip install coverage`."
    ) from exc

try:
    import pytest  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("pytest is required to run the click tests.") from exc


@dataclass(frozen=True)
class FunctionInfo:
    """Metadata for a single function or method."""

    module: str
    qualname: str
    filepath: Path
    start: int
    end: int

    @property
    def key(self) -> str:
        """Return the canonical lookup key for this function."""
        return f"{self.module}:{self.qualname}"


@dataclass
class FileFunctionIndex:
    """Line-to-function lookup for a source file."""

    functions: List[FunctionInfo]
    lines_to_functions: Dict[int, List[FunctionInfo]]


class FunctionCollector(ast.NodeVisitor):
    """Collect nested function and method definitions with their line ranges."""

    def __init__(self, module: str, filepath: Path) -> None:
        """Store module metadata and prep state for a traversal."""
        self.module = module
        self.filepath = filepath.resolve()
        self._scope: List[str] = []
        self.functions: List[FunctionInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class scope so nested method names get the right prefixes."""
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record synchronous function definitions."""
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record async function definitions."""
        self._record_function(node)

    def _record_function(self, node: ast.AST) -> None:
        """Convert an AST node into FunctionInfo entries (including nested scopes)."""
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return
        qualname = ".".join(self._scope + [node.name])
        end_lineno = getattr(node, "end_lineno", None)
        if end_lineno is None:
            end_lineno = self._infer_end_lineno(node)
        info = FunctionInfo(
            module=self.module,
            qualname=qualname,
            filepath=self.filepath,
            start=node.lineno,
            end=end_lineno,
        )
        self.functions.append(info)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    @staticmethod
    def _infer_end_lineno(node: ast.AST) -> int:
        """Best-effort fallback when the parser didn't record an end line."""
        max_lineno = getattr(node, "lineno", 0)
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                max_lineno = max(max_lineno, getattr(child, "lineno"))
        return max_lineno


class CoverageGraph:
    """Convert coverage data with per-test contexts into a tree."""

    def __init__(
        self,
        src_root: Path,
        file_index: Mapping[Path, FileFunctionIndex],
        lookup: Mapping[str, FunctionInfo],
    ) -> None:
        """Initialize the graph with source metadata and empty adjacency lists."""
        self.src_root = src_root
        self.file_index = file_index
        self.lookup = lookup
        self.function_to_tests: Dict[str, Set[str]] = defaultdict(set)
        self.test_to_functions: Dict[str, Set[str]] = defaultdict(set)

    def merge(self, data: coverage.CoverageData) -> None:
        """Populate adjacency lists by replaying coverage line→context data."""
        if not hasattr(data, "contexts_by_lineno"):
            raise SystemExit(
                "Coverage was run without per-test contexts. Upgrade coverage>=5."
            )
        for filename in data.measured_files():
            path = Path(filename).resolve()
            try:
                path.relative_to(self.src_root)
            except ValueError:
                continue
            index = self.file_index.get(path)
            if not index:
                continue
            contexts_map = data.contexts_by_lineno(filename)
            if not contexts_map:
                continue
            for line_no, contexts in contexts_map.items():
                funcs = index.lines_to_functions.get(line_no)
                if not funcs:
                    continue
                for context in contexts:
                    if not context or not _looks_like_test_context(context):
                        continue
                    for func in funcs:
                        self.function_to_tests[func.key].add(context)
                        self.test_to_functions[context].add(func.key)

    def build_tree(self) -> Dict[str, MutableMapping[str, object]]:
        """Return a nested dict module→class→method with attached test leaves."""
        tree: Dict[str, MutableMapping[str, object]] = {}
        for func_key, tests in self.function_to_tests.items():
            module, qual = func_key.split(":", 1)
            node = tree.setdefault(module, {})
            for part in qual.split("."):
                node = node.setdefault(part, {})  # type: ignore[assignment]
            node.setdefault("__tests__", set())  # type: ignore[index]
            node["__tests__"].update(tests)  # type: ignore[index]

        def finalize(subtree: MutableMapping[str, object]) -> None:
            """Convert set leaves to sorted lists for reproducible output."""
            tests = subtree.get("__tests__")
            if isinstance(tests, set):
                subtree["__tests__"] = sorted(tests)
            for key, child in list(subtree.items()):
                if key == "__tests__":
                    continue
                if isinstance(child, MutableMapping):
                    finalize(child)

        for module_name, module_tree in tree.items():
            if isinstance(module_tree, MutableMapping):
                finalize(module_tree)
        return tree

    def print_tree(self, tree: Mapping[str, object]) -> None:
        """Pretty-print tree nodes using indentation."""
        if not tree:
            print("No methods were executed by the tests.")
            return
        for module in sorted(tree):
            print(module)
            node = tree[module]
            if isinstance(node, Mapping):
                self._print_subtree(node, indent=1)

    def _print_subtree(self, subtree: Mapping[str, object], indent: int) -> None:
        """Recursive helper for print_tree that handles child nodes and leaves."""
        pad = "  " * indent
        child_keys = sorted(key for key in subtree.keys() if key != "__tests__")
        for key in child_keys:
            print(f"{pad}{key}")
            child = subtree[key]
            if isinstance(child, Mapping):
                self._print_subtree(child, indent + 1)
        tests = subtree.get("__tests__")
        if isinstance(tests, list):
            for test in tests:
                print(f"{pad}- [test] {test}")

    def print_method_matches(self, pattern: str, fuzzy: bool = False) -> None:
        """Display tests associated with matching methods."""
        matches = self.find_method_matches(pattern, fuzzy=fuzzy)
        if not matches:
            scope = "substring" if fuzzy else "exact"
            print(f"No {scope} matches found for '{pattern}'.")
            return
        for method in sorted(matches):
            info = self.lookup.get(method)
            location = ""
            if info:
                rel = info.filepath.relative_to(self.src_root)
                location = f" ({rel}:{info.start})"
            print(f"{method}{location}")
            for test in sorted(self.function_to_tests.get(method, [])):
                print(f"  - {test}")

    def find_method_matches(self, pattern: str, fuzzy: bool = False) -> Set[str]:
        """Return set of method keys matching either exactly or by substring."""
        if not fuzzy:
            return {pattern} if pattern in self.function_to_tests else set()
        needle = pattern.lower()
        return {
            method
            for method in self.function_to_tests
            if needle in method.lower()
        }

    def export_json(self, path: Path) -> None:
        """Serialize method/test adjacency info for downstream tooling."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "functions": {
                method: {
                    "file": str(info.filepath),
                    "relative_file": str(info.filepath.relative_to(self.src_root)),
                    "line_range": [info.start, info.end],
                    "tests": sorted(self.function_to_tests[method]),
                }
                for method, info in self.lookup.items()
                if method in self.function_to_tests
            },
            "tests": {
                test: sorted(functions)
                for test, functions in self.test_to_functions.items()
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote JSON graph to {path}")


def _looks_like_test_context(name: str) -> bool:
    """Heuristically detect pytest test nodeids (skip internal coverage contexts)."""
    return "::" in name and not name.startswith("pytest:")


def build_function_index(src_root: Path) -> tuple[Dict[Path, FileFunctionIndex], Dict[str, FunctionInfo]]:
    """Walk the source tree and build lookup tables for every function definition."""
    file_index: Dict[Path, FileFunctionIndex] = {}
    lookup: Dict[str, FunctionInfo] = {}
    for py_file in src_root.rglob("*.py"):
        if py_file.name.startswith("."):
            continue
        module = ".".join(py_file.relative_to(src_root).with_suffix("").parts)
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError) as exc:
            print(f"Skipping {py_file}: {exc}", file=sys.stderr)
            continue
        collector = FunctionCollector(module, py_file)
        collector.visit(tree)
        if not collector.functions:
            continue
        line_map: Dict[int, List[FunctionInfo]] = defaultdict(list)
        for func in collector.functions:
            lookup[func.key] = func
            for line in range(func.start, func.end + 1):
                line_map[line].append(func)
        file_index[collector.filepath] = FileFunctionIndex(
            functions=collector.functions,
            lines_to_functions=line_map,
        )
    if not file_index:
        raise SystemExit(f"No Python files discovered under {src_root}.")
    return file_index, lookup


class CoverageContextPlugin:
    """Pytest plugin that switches coverage contexts per test."""

    def __init__(self, cov: coverage.Coverage):
        """Store the coverage object we need to drive from pytest hooks."""
        self.coverage = cov

    @pytest.hookimpl(tryfirst=True, hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):  # type: ignore[override]
        """Switch coverage contexts before/after each test case."""
        del nextitem  # unused
        self.coverage.switch_context(item.nodeid)
        yield
        self.coverage.switch_context("pytest:idle")


@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily change directories."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Configure CLI arguments for controlling coverage execution and queries."""
    parser = argparse.ArgumentParser(
        description=(
            "Run pytest with coverage contexts and build a tree that links click "
            "methods to the tests that execute them."
        )
    )
    parser.add_argument(
        "--project-name",
        default="click",
        type=str
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default= "../project",
        help="Path to the click project. Defaults to ../project relative to this file.",
    )
    parser.add_argument(
        "--tests-path",
        default=".",
        help="Path (relative to project root) where pytest tests live. Default: tests",
    )
    parser.add_argument(
        "--src-path",
        default="src",
        help="Implementation source root relative to the project root. Default: src",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        dest="pytest_args",
        help="Extra argument to forward to pytest. Specify multiple times as needed.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Continue building the graph even if pytest exits with a non-zero status.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Fully qualified method key (module:qualname) to list connected tests for.",
    )
    parser.add_argument(
        "--method-contains",
        action="append",
        help="Substring to fuzzy-match against method keys when listing tests.",
    )
    parser.add_argument(
        "--skip-tree",
        action="store_true",
        help="Skip printing the ASCII tree (useful when only querying methods).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: orchestrate coverage run, graph building, and reporting."""
    args = parse_args(argv)
    project_root =  args.project_root / args.project_name
    if not project_root.exists():
        raise SystemExit(f"Project root {project_root} does not exist.")
    src_root = (project_root / args.src_path).resolve()
    tests_path = Path(args.tests_path)
    output_dir = Path("output") / args.project_name
    os.makedirs(output_dir, exist_ok=True)
    output_json = output_dir / "function_testunit_mapping.json"

    sys.path.insert(0, str(src_root))
    sys.path.insert(0, str(project_root))

    file_index, lookup = build_function_index(src_root)
    cov = coverage.Coverage(branch=True, source=[str(src_root)])
    cov.erase()
    plugin = CoverageContextPlugin(cov)

    pytest_args = [str(tests_path)]
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    exit_code = 0
    with pushd(project_root):
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=".", check=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=".", check=True)
        subprocess.run(['pip', 'install', '-e', '.'], cwd=".", check=True)
        cov.start()
        try:
            exit_code = pytest.main(pytest_args, plugins=[plugin])
        finally:
            cov.stop()
            cov.save()

    if exit_code != 0:
        message = f"pytest exited with status {exit_code}."
        if not args.allow_failures:
            raise SystemExit(message)
        print(f"Warning: {message} Proceeding with partial data.", file=sys.stderr)

    data = cov.get_data()
    graph = CoverageGraph(src_root, file_index, lookup)
    graph.merge(data)
    tree = graph.build_tree()

    print(
        f"Mapped {len(graph.function_to_tests)} methods across "
        f"{len(graph.test_to_functions)} tests."
    )

    graph.export_json(output_json)

    # queries_ran = False
    # if args.method:
    #     for method in args.method:
    #         graph.print_method_matches(method, fuzzy=False)
    #     queries_ran = True
    # if args.method_contains:
    #     for pattern in args.method_contains:
    #         graph.print_method_matches(pattern, fuzzy=True)
    #     queries_ran = True

    # if not args.skip_tree and not queries_ran:
    #     graph.print_tree(tree)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
