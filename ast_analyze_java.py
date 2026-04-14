#!/usr/bin/env python
"""
Build a method-to-test mapping for Java projects using javalang AST analysis
and JaCoCo coverage data.

Per-test-class granularity: each test class is run separately and its JaCoCo
exec file is parsed to determine which source methods are covered.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import javalang
except ImportError as exc:
    raise SystemExit(
        "The javalang package is required. Install it with `pip install javalang`."
    ) from exc

from utils import pushd, get_spec


# ---------------------------------------------------------------------------
# Data classes (mirrors Python version's FunctionInfo / ClassInfo)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JavaMethodInfo:
    """Metadata for a single Java method or constructor."""
    package: str
    qualname: str       # e.g. "FileUtils.copyFile" or "FileUtils.InnerClass.foo"
    filepath: Path
    start: int
    end: int
    return_type: str
    parameter_types: List[Tuple[str, str]]   # [(name, type), ...]
    modifiers: frozenset

    @property
    def key(self) -> str:
        return f"{self.package}.{self.qualname}"


@dataclass(frozen=True)
class JavaClassInfo:
    """Metadata for a discovered Java class/interface/enum."""
    package: str
    qualname: str
    extends: Optional[str]
    implements: List[str]
    modifiers: frozenset

    @property
    def key(self) -> str:
        return f"{self.package}.{self.qualname}"


@dataclass
class JavaFileIndex:
    """Line-to-method lookup for a single Java source file."""
    methods: List[JavaMethodInfo]
    lines_to_methods: Dict[int, List[JavaMethodInfo]]
    classes: List[JavaClassInfo]


# ---------------------------------------------------------------------------
# Java AST analysis using javalang
# ---------------------------------------------------------------------------

def _compute_end_line(source_lines: List[str], start_line: int) -> int:
    """Estimate the end line of a method by counting braces from the start line."""
    depth = 0
    found_open = False
    for i in range(start_line - 1, len(source_lines)):
        line = source_lines[i]
        # Strip string literals and comments to avoid counting braces inside them
        stripped = _strip_strings_and_comments(line)
        for ch in stripped:
            if ch == '{':
                depth += 1
                found_open = True
            elif ch == '}':
                depth -= 1
                if found_open and depth == 0:
                    return i + 1  # 1-indexed
    return start_line


def _strip_strings_and_comments(line: str) -> str:
    """Roughly strip string literals and line comments from a line."""
    result = []
    i = 0
    in_string = False
    quote_char = None
    while i < len(line):
        ch = line[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == quote_char:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
            i += 1
            continue
        if ch == '/' and i + 1 < len(line):
            if line[i + 1] == '/':
                break  # rest is comment
            if line[i + 1] == '*':
                # skip block comment on this line
                end = line.find('*/', i + 2)
                if end != -1:
                    i = end + 2
                    continue
                else:
                    break
        result.append(ch)
        i += 1
    return ''.join(result)


_JAVA_PRIMITIVES = frozenset({
    "void", "boolean", "byte", "char", "short", "int", "long", "float", "double",
})

_JAVA_LANG_TYPES = frozenset({
    "Object", "String", "Class", "System", "Thread", "Throwable", "Exception",
    "RuntimeException", "Error", "Integer", "Long", "Double", "Float", "Boolean",
    "Byte", "Short", "Character", "Number", "Math", "StringBuilder", "StringBuffer",
    "Comparable", "Iterable", "Cloneable", "Override", "Deprecated", "SuppressWarnings",
    "Enum", "Void", "Process", "ProcessBuilder", "Runtime", "StackTraceElement",
    "ClassLoader", "Package", "SecurityManager", "AutoCloseable",
})


def _resolve_type_name(
    type_node,
    imports: Dict[str, str],
    package: str,
    package_prefix: str,
) -> str:
    """Resolve a javalang type reference to a qualified name."""
    if type_node is None:
        return "void"
    if isinstance(type_node, str):
        return type_node
    name = getattr(type_node, 'name', None)
    if name is None:
        return str(type_node)
    # Primitives stay as-is
    if name in _JAVA_PRIMITIVES:
        return name
    # java.lang types
    if name in _JAVA_LANG_TYPES:
        return name
    # Explicitly imported
    if name in imports:
        return imports[name]
    # In the same package (only for project types)
    if package.startswith(package_prefix):
        return f"{package}.{name}"
    return name


def _collect_imports(tree) -> Dict[str, str]:
    """Collect import alias -> fully qualified name mapping."""
    imports: Dict[str, str] = {}
    if tree.imports:
        for imp in tree.imports:
            path = imp.path
            if imp.static:
                # static import: e.g. import static java.util.Collections.sort
                parts = path.rsplit('.', 1)
                if len(parts) == 2:
                    imports[parts[1]] = path
            else:
                # e.g. import java.util.List -> List -> java.util.List
                simple_name = path.rsplit('.', 1)[-1]
                if simple_name != '*':
                    imports[simple_name] = path
    return imports


def parse_java_file(
    filepath: Path,
    package_prefix: str,
) -> Optional[Tuple[List[JavaMethodInfo], List[JavaClassInfo]]]:
    """Parse a single .java file and extract method/class metadata."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = javalang.parse.parse(source)
    except Exception as exc:
        print(f"Skipping {filepath}: {exc}", file=sys.stderr)
        return None

    source_lines = source.splitlines()
    package = tree.package.name if tree.package else ""
    imports = _collect_imports(tree)

    methods: List[JavaMethodInfo] = []
    classes: List[JavaClassInfo] = []

    # Collect classes/interfaces/enums
    for cls_type in (javalang.tree.ClassDeclaration,
                     javalang.tree.InterfaceDeclaration,
                     javalang.tree.EnumDeclaration):
        for path_nodes, node in tree.filter(cls_type):
            scope = [
                n.name for n in path_nodes
                if hasattr(n, 'name') and isinstance(
                    n, (javalang.tree.ClassDeclaration,
                        javalang.tree.InterfaceDeclaration,
                        javalang.tree.EnumDeclaration))
            ]
            qualname = ".".join(scope + [node.name])

            extends = None
            if hasattr(node, 'extends') and node.extends:
                if isinstance(node.extends, list):
                    extends = ", ".join(
                        _resolve_type_name(e, imports, package, package_prefix)
                        for e in node.extends
                    )
                else:
                    extends = _resolve_type_name(
                        node.extends, imports, package, package_prefix
                    )

            impl_list = []
            if hasattr(node, 'implements') and node.implements:
                impl_list = [
                    _resolve_type_name(i, imports, package, package_prefix)
                    for i in node.implements
                ]

            mods = frozenset(node.modifiers) if node.modifiers else frozenset()
            classes.append(JavaClassInfo(
                package=package,
                qualname=qualname,
                extends=extends,
                implements=impl_list,
                modifiers=mods,
            ))

    # Collect methods and constructors
    for decl_type in (javalang.tree.MethodDeclaration,
                      javalang.tree.ConstructorDeclaration):
        for path_nodes, node in tree.filter(decl_type):
            if node.position is None:
                continue
            scope = [
                n.name for n in path_nodes
                if hasattr(n, 'name') and isinstance(
                    n, (javalang.tree.ClassDeclaration,
                        javalang.tree.InterfaceDeclaration,
                        javalang.tree.EnumDeclaration))
            ]
            qualname = ".".join(scope + [node.name])
            start_line = node.position.line
            end_line = _compute_end_line(source_lines, start_line)

            # Return type
            if isinstance(node, javalang.tree.ConstructorDeclaration):
                ret_type = "<init>"
            else:
                ret_type = _resolve_type_name(
                    node.return_type, imports, package, package_prefix
                )

            # Parameters
            params = []
            if node.parameters:
                for p in node.parameters:
                    ptype = _resolve_type_name(
                        p.type, imports, package, package_prefix
                    )
                    params.append((p.name, ptype))

            mods = frozenset(node.modifiers) if node.modifiers else frozenset()
            methods.append(JavaMethodInfo(
                package=package,
                qualname=qualname,
                filepath=filepath.resolve(),
                start=start_line,
                end=end_line,
                return_type=ret_type,
                parameter_types=params,
                modifiers=mods,
            ))

    return methods, classes


def build_java_function_index(
    src_root: Path,
    package_prefix: str,
) -> Tuple[
    Dict[Path, JavaFileIndex],
    Dict[str, JavaMethodInfo],
    Dict[str, JavaClassInfo],
]:
    """Walk the Java source tree and build lookup tables."""
    file_index: Dict[Path, JavaFileIndex] = {}
    method_lookup: Dict[str, JavaMethodInfo] = {}
    class_lookup: Dict[str, JavaClassInfo] = {}

    for java_file in sorted(src_root.rglob("*.java")):
        if java_file.name.startswith("."):
            continue
        result = parse_java_file(java_file, package_prefix)
        if result is None:
            continue
        methods, classes = result
        if not methods and not classes:
            continue

        line_map: Dict[int, List[JavaMethodInfo]] = defaultdict(list)
        for m in methods:
            method_lookup[m.key] = m
            for line in range(m.start, m.end + 1):
                line_map[line].append(m)

        for c in classes:
            class_lookup[c.key] = c

        resolved = java_file.resolve()
        file_index[resolved] = JavaFileIndex(
            methods=methods,
            lines_to_methods=line_map,
            classes=classes,
        )

    if not file_index:
        raise SystemExit(f"No Java files discovered under {src_root}.")
    return file_index, method_lookup, class_lookup


# ---------------------------------------------------------------------------
# JaCoCo coverage collection and parsing
# ---------------------------------------------------------------------------

def _discover_test_classes(test_root: Path) -> List[str]:
    """Find all test class FQNs under the test source root."""
    test_classes = []
    for java_file in sorted(test_root.rglob("*Test*.java")):
        if java_file.name.startswith("."):
            continue
        # Skip abstract test classes and inner classes
        if "$" in java_file.name:
            continue
        rel = java_file.relative_to(test_root)
        # Convert path to FQN: org/apache/commons/io/FileUtilsTest.java -> org.apache.commons.io.FileUtilsTest
        fqn = str(rel.with_suffix("")).replace(os.sep, ".")
        test_classes.append(fqn)
    return test_classes


def _run_maven_test_with_jacoco(
    project_root: Path,
    test_class: str,
    exec_dir: Path,
) -> bool:
    """Run a single test class with JaCoCo coverage, return True if successful."""
    exec_file = exec_dir / f"{test_class}.exec"
    if exec_file.exists():
        return True

    jacoco_agent_arg = (
        f"-javaagent:${{settings.localRepository}}/org/jacoco/org.jacoco.agent/"
        f"0.8.12/org.jacoco.agent-0.8.12-runtime.jar=destfile={exec_file}"
    )

    cmd = [
        "mvn", "test",
        f"-Dtest={test_class}",
        "-DfailIfNoTests=false",
        "-Dmaven.test.failure.ignore=true",
        f"-Djacoco.destFile={exec_file}",
        "-pl", ".",
        "-q",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return exec_file.exists()
    except subprocess.TimeoutExpired:
        print(f"  Timeout running {test_class}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"  Error running {test_class}: {exc}", file=sys.stderr)
        return False


def _run_all_tests_with_jacoco(
    project_root: Path,
    exec_dir: Path,
) -> bool:
    """Run all tests at once with JaCoCo, producing a single exec file."""
    exec_file = exec_dir / "jacoco-all.exec"

    cmd = [
        "mvn", "test",
        "-Dmaven.test.failure.ignore=true",
        f"-Djacoco.destFile={exec_file}",
        "-pl", ".",
    ]

    print("Running all tests with JaCoCo coverage...")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"Warning: mvn test exited with code {result.returncode}", file=sys.stderr)
        return exec_file.exists()
    except subprocess.TimeoutExpired:
        print("Timeout running tests", file=sys.stderr)
        return False


def _run_per_test_class_coverage(
    project_root: Path,
    test_root: Path,
    exec_dir: Path,
) -> Dict[str, Path]:
    """Run each test class separately with JaCoCo. Returns test_class -> exec_file."""
    test_classes = _discover_test_classes(test_root)
    print(f"Discovered {len(test_classes)} test classes")

    exec_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    for i, tc in enumerate(test_classes, 1):
        exec_file = exec_dir / f"{tc}.exec"
        simple_name = tc.rsplit(".", 1)[-1]
        print(f"  [{i}/{len(test_classes)}] Testing {simple_name}...", end="", flush=True)

        cmd = [
            "mvn", "test",
            f"-Dtest={simple_name}",
            "-DfailIfNoTests=false",
            "-Dmaven.test.failure.ignore=true",
            f"-Djacoco.destFile={exec_file}",
            "-pl", ".",
            "-q",
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            print(" TIMEOUT")
            continue
        except Exception as exc:
            print(f" ERROR: {exc}")
            continue

        if exec_file.exists():
            results[tc] = exec_file
            print(" OK")
        else:
            print(" no coverage")

    print(f"Collected coverage for {len(results)}/{len(test_classes)} test classes")
    return results


def _generate_jacoco_xml(
    project_root: Path,
    exec_file: Path,
    xml_output: Path,
    src_root: Path,
    classes_dir: Path,
) -> bool:
    """Generate JaCoCo XML report from an exec file using jacoco:report."""
    cmd = [
        "mvn", "jacoco:report",
        f"-Djacoco.dataFile={exec_file}",
        f"-Djacoco.outputDirectory={xml_output.parent}",
        "-pl", ".",
        "-q",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        report_xml = xml_output.parent / "jacoco.xml"
        return report_xml.exists()
    except Exception:
        return False


def _parse_jacoco_xml(xml_path: Path) -> Dict[str, Set[int]]:
    """Parse a JaCoCo XML report and return source_file -> set of covered lines.

    Returns a dict mapping 'package/SourceFile.java' -> {covered line numbers}.
    """
    covered_lines: Dict[str, Set[int]] = defaultdict(set)
    try:
        tree_xml = ET.parse(str(xml_path))
    except Exception as exc:
        print(f"Error parsing {xml_path}: {exc}", file=sys.stderr)
        return covered_lines

    root = tree_xml.getroot()
    for pkg in root.iter("package"):
        pkg_name = pkg.get("name", "")  # e.g. "org/apache/commons/io"
        for srcfile in pkg.iter("sourcefile"):
            src_name = srcfile.get("name", "")  # e.g. "FileUtils.java"
            src_key = f"{pkg_name}/{src_name}" if pkg_name else src_name
            for line_elem in srcfile.iter("line"):
                nr = int(line_elem.get("nr", "0"))
                ci = int(line_elem.get("ci", "0"))  # covered instructions
                if ci > 0:
                    covered_lines[src_key].add(nr)

    return covered_lines


# ---------------------------------------------------------------------------
# Coverage graph: merge JaCoCo data with AST index
# ---------------------------------------------------------------------------

class JavaCoverageGraph:
    """Map JaCoCo coverage data to Java method AST metadata."""

    def __init__(
        self,
        src_root: Path,
        file_index: Dict[Path, JavaFileIndex],
        method_lookup: Dict[str, JavaMethodInfo],
        class_lookup: Dict[str, JavaClassInfo],
    ) -> None:
        self.src_root = src_root
        self.file_index = file_index
        self.method_lookup = method_lookup
        self.class_lookup = class_lookup
        self.method_to_tests: Dict[str, Set[str]] = defaultdict(set)

    def merge_per_class(
        self,
        test_class: str,
        covered_lines: Dict[str, Set[int]],
    ) -> None:
        """Merge coverage from a single test class execution."""
        for src_key, lines in covered_lines.items():
            # src_key is like "org/apache/commons/io/FileUtils.java"
            # Find the matching file in our index
            matching_file = self._find_source_file(src_key)
            if matching_file is None:
                continue
            index = self.file_index.get(matching_file)
            if index is None:
                continue
            for line_no in lines:
                methods = index.lines_to_methods.get(line_no)
                if not methods:
                    continue
                for method in methods:
                    self.method_to_tests[method.key].add(test_class)

    def merge_aggregate(
        self,
        covered_lines: Dict[str, Set[int]],
    ) -> None:
        """Merge aggregate coverage (no per-test distinction)."""
        self.merge_per_class("__aggregate__", covered_lines)

    def _find_source_file(self, src_key: str) -> Optional[Path]:
        """Find a resolved filepath matching a JaCoCo source key."""
        # src_key: "org/apache/commons/io/FileUtils.java"
        # We need to find this file under src_root
        candidate = (self.src_root / src_key).resolve()
        if candidate in self.file_index:
            return candidate
        # Fallback: search by suffix
        for fpath in self.file_index:
            if str(fpath).endswith(src_key.replace("/", os.sep)):
                return fpath
        return None

    def export_json(self, path: Path, meta: dict = {}, project_root: Optional[Path] = None) -> None:
        """Serialize method/test mapping to JSON (same format as Python version)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rel_base = project_root.resolve() if project_root else self.src_root.parent

        functions = {}
        for method_key, info in self.method_lookup.items():
            tests = sorted(self.method_to_tests.get(method_key, []))
            if not tests:
                continue
            try:
                rel_file = str(info.filepath.relative_to(rel_base))
            except ValueError:
                rel_file = str(info.filepath)

            functions[method_key] = {
                "file": str(info.filepath),
                "relative_file": rel_file,
                "line_range": [info.start, info.end],
                "tests": tests,
                "return_type": info.return_type,
                "parameter_types": [
                    {"name": name, "type": ptype}
                    for name, ptype in info.parameter_types
                ],
                "modifiers": sorted(info.modifiers),
                "variable_types": {},
            }

        payload = {
            "meta": meta,
            "functions": functions,
        }

        if self.class_lookup:
            payload["classes"] = {
                key: {
                    "package": info.package,
                    "qualname": info.qualname,
                    "extends": info.extends,
                    "implements": info.implements,
                    "modifiers": sorted(info.modifiers),
                }
                for key, info in self.class_lookup.items()
            }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON graph to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_java_function_mapping(
    project_name: str,
    project_path: str = "../project",
) -> Optional[int]:
    """Main entry point: AST analysis + JaCoCo coverage -> JSON mapping."""
    repo_spec = get_spec(project_name)
    if not repo_spec:
        print(f"Project {project_name} not found in repo_list.json", file=sys.stderr)
        return 1

    project_root = Path(project_path) / project_name
    if not project_root.exists():
        raise SystemExit(f"Project root {project_root} does not exist.")

    src_path = repo_spec.get("src_path", "src/main/java")
    test_path = repo_spec.get("test_path", "src/test/java")
    package_prefix = repo_spec.get("package_prefix", "")
    commit_id = repo_spec.get("commit_id", "")

    src_root = (project_root / src_path).resolve()
    test_root = (project_root / test_path).resolve()

    output_dir = Path("output") / project_name
    os.makedirs(output_dir, exist_ok=True)
    output_json = output_dir / "function_testunit_mapping.json"

    if output_json.exists():
        print(f"Output file {output_json} already exists.")
        return 0

    # Step 1: Reset to target commit
    print(f"Resetting to commit {commit_id[:12]}...")
    with pushd(project_root):
        subprocess.run(["git", "reset", "--hard", commit_id], check=True)
        subprocess.run(["git", "clean", "-fd"], check=True)

    # Step 2: Build project (compile only, skip tests)
    print("Compiling project...")
    build_cmd = repo_spec.get("build_cmd", "mvn compile -DskipTests")
    if isinstance(build_cmd, str):
        build_cmd = [build_cmd]
    for cmd in build_cmd:
        subprocess.run(cmd.split(), cwd=str(project_root), check=True)

    # Step 3: AST analysis
    print(f"Analyzing Java source in {src_root}...")
    file_index, method_lookup, class_lookup = build_java_function_index(
        src_root, package_prefix
    )
    print(
        f"Found {len(method_lookup)} methods across "
        f"{len(file_index)} files, {len(class_lookup)} classes"
    )

    # Step 4: Run tests with JaCoCo (per-test-class)
    exec_dir = (project_root / "target" / "jacoco-per-test").resolve()
    test_coverage = _run_per_test_class_coverage(
        project_root.resolve(), test_root, exec_dir
    )

    # Step 5: Parse coverage and build graph
    print("Parsing coverage data...")
    graph = JavaCoverageGraph(src_root, file_index, method_lookup, class_lookup)

    report_dir = project_root / "target" / "site" / "jacoco"
    jacoco_xml = report_dir / "jacoco.xml"

    for test_class, exec_file in test_coverage.items():
        # Generate XML report for this exec file
        cmd = [
            "mvn", "jacoco:report",
            f"-Djacoco.dataFile={exec_file}",
            "-pl", ".",
            "-q",
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception:
            continue

        if jacoco_xml.exists():
            covered = _parse_jacoco_xml(jacoco_xml)
            simple_name = test_class.rsplit(".", 1)[-1]
            graph.merge_per_class(simple_name, covered)

    # Step 6: Export
    meta = {
        "src_path": src_path,
        "test_path": test_path,
        "commit_id": commit_id,
        "language": "java",
        "package_prefix": package_prefix,
        "coverage_granularity": "per-test-class",
    }
    graph.export_json(output_json, meta, project_root=project_root.resolve())

    mapped_count = sum(1 for tests in graph.method_to_tests.values() if tests)
    total_tests = set()
    for tests in graph.method_to_tests.values():
        total_tests.update(tests)
    print(
        f"Mapped {mapped_count} methods across "
        f"{len(total_tests)} test classes."
    )
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build method-to-test mapping for Java projects."
    )
    parser.add_argument(
        "--project-name",
        default="commons-io",
        type=str,
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="../project",
        help="Path to the project directory.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(generate_java_function_mapping(args.project_name, args.project_root))
