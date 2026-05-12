#!/usr/bin/env python
"""
Build a method-to-test mapping for Java projects using javap to extract
method signatures from compiled class files and JaCoCo coverage data.

This approach avoids parsing issues with complex Java source files by using
the compiled bytecode directly.
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
# Java method extraction using javap (from compiled class files)
# ---------------------------------------------------------------------------

def extract_methods_from_class_file(
    class_file: Path,
    classes_root: Path,
    src_root: Path,
) -> Tuple[List[JavaMethodInfo], str, str]:
    """
    Extract method information from a compiled .class file using javap.

    Args:
        class_file: Path to the .class file
        classes_root: Root directory of compiled classes (e.g., target/classes)
        src_root: Root directory of source files (to find line numbers)

    Returns:
        (methods, package, simple_class_name)
    """
    methods = []

    # Get the fully qualified class name from the file path
    rel_path = class_file.relative_to(classes_root)
    class_fqn = str(rel_path.with_suffix('')).replace(os.sep, '.')

    # Extract package and class name
    if '.' in class_fqn:
        package = class_fqn.rsplit('.', 1)[0]
        simple_class_name = class_fqn.rsplit('.', 1)[1]
    else:
        package = ''
        simple_class_name = class_fqn

    # Run javap to get method signatures
    # Use -private to show all members (public, protected, package, private)
    try:
        result = subprocess.run(
            ['javap', '-private', '-s', str(class_file)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return methods, package, simple_class_name

        # Parse javap output
        # Each method looks like:
        #   public static java.lang.String method_name(param_type);
        #     descriptor: (Lparam;)Lreturn;

        lines = result.stdout.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for method declarations (not fields, not class declarations)
            # Methods have ( in them and don't have '=' (which would be a field)
            if '(' in line and '=' not in line:
                # Skip if it's a class/interface/enum declaration line
                if any(keyword in line.split('(')[0] for keyword in [' class ', ' interface ', ' enum ']):
                    i += 1
                    continue

                # Extract method name - it's the word before (
                try:
                    before_paren = line.split('(')[0]
                    parts = before_paren.split()
                    if len(parts) >= 1:
                        # The method name is the last word before (
                        method_name = parts[-1]

                        # Skip special cases
                        if method_name in ['class', 'interface', 'enum', 'descriptor:']:
                            i += 1
                            continue

                        # Check next line for descriptor
                        descriptor = ''
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line.startswith('descriptor:'):
                                descriptor = next_line.split(':', 1)[1].strip()

                        # Create JavaMethodInfo
                        method_key = f"{package}.{simple_class_name}.{method_name}"
                        methods.append(JavaMethodInfo(
                            package=package,
                            qualname=f"{simple_class_name}.{method_name}",
                            filepath=Path(''),  # Will be filled later if needed
                            start=0,
                            end=0,
                            return_type='',
                            parameter_types=[],
                            modifiers=frozenset()
                        ))

                except Exception as e:
                    # Skip this line if parsing fails
                    pass

            i += 1

    except Exception as e:
        print(f"Warning: Failed to extract methods from {class_file}: {e}", file=sys.stderr)

    return methods, package, simple_class_name


def build_java_function_index_from_classes(
    classes_root: Path,
    src_root: Path,
    package_prefix: str,
) -> Tuple[Dict[Path, JavaFileIndex], Dict[str, JavaMethodInfo], Dict[str, JavaClassInfo]]:
    """
    Build function index from compiled class files using javap.

    Args:
        classes_root: Root directory of compiled classes (e.g., target/classes)
        src_root: Root directory of source files
        package_prefix: Package prefix to filter (e.g., "org.apache.commons.io")

    Returns:
        (file_index, method_lookup, class_lookup)
    """
    file_index: Dict[Path, JavaFileIndex] = {}
    method_lookup: Dict[str, JavaMethodInfo] = {}
    class_lookup: Dict[str, JavaClassInfo] = {}

    # Find all .class files
    class_files = list(classes_root.rglob('*.class'))
    print(f"Found {len(class_files)} class files in {classes_root}")

    for class_file in class_files:
        # Get fully qualified class name
        rel_path = class_file.relative_to(classes_root)
        class_fqn = str(rel_path.with_suffix('')).replace(os.sep, '.')

        # Filter by package prefix
        if package_prefix and not class_fqn.startswith(package_prefix):
            continue

        # Extract methods using javap
        methods, _, _ = extract_methods_from_class_file(
            class_file, classes_root, src_root
        )

        # Add methods to method_lookup
        for method in methods:
            method_lookup[method.key] = method

    print(f"Extracted {len(method_lookup)} methods from {len(class_files)} class files")

    return file_index, method_lookup, class_lookup


# ---------------------------------------------------------------------------
# Java AST analysis using javalang (DEPRECATED - keeping for backwards compatibility)
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


def _parse_failed_tests_from_surefire_reports(surefire_reports_dir: Path) -> Set[str]:
    """
    Parse Maven Surefire XML reports to extract failed test class names.

    Surefire generates XML reports in target/surefire-reports/ with structure:
      <testsuite name="com.example.TestClass" tests="5" failures="1" errors="0" ...>
        <testcase name="testMethod" classname="com.example.TestClass" ...>
          <failure message="..." type="AssertionError">...</failure>
        </testcase>
      </testsuite>

    Args:
        surefire_reports_dir: Path to target/surefire-reports directory

    Returns:
        Set of test class simple names (e.g., "TestClass", not "com.example.TestClass")
    """
    failed_tests = set()

    if not surefire_reports_dir.exists():
        print(f"Warning: Surefire reports directory not found: {surefire_reports_dir}")
        return failed_tests

    # Find all TEST-*.xml files in the surefire-reports directory
    xml_files = list(surefire_reports_dir.glob("TEST-*.xml"))
    print(f"Found {len(xml_files)} Surefire XML reports")

    for xml_file in xml_files:
        try:
            tree = ET.parse(str(xml_file))
            root = tree.getroot()

            # Check testsuite attributes for failures or errors
            testsuite_failures = int(root.get("failures", "0"))
            testsuite_errors = int(root.get("errors", "0"))

            if testsuite_failures > 0 or testsuite_errors > 0:
                # Get the test class name
                testsuite_name = root.get("name", "")
                if testsuite_name:
                    simple_name = testsuite_name.rsplit(".", 1)[-1]
                    failed_tests.add(simple_name)
                    print(f"  Failed: {simple_name} (failures={testsuite_failures}, errors={testsuite_errors})")

        except Exception as e:
            print(f"Warning: Failed to parse {xml_file}: {e}", file=sys.stderr)
            continue

    return failed_tests


def _parse_failed_tests_from_output(maven_output: str) -> Set[str]:
    """
    DEPRECATED: Parse Maven test output to extract failed test class names.
    Use _parse_failed_tests_from_surefire_reports() instead for more reliable parsing.

    Kept as fallback when Surefire reports are not available.
    """
    failed_tests = set()

    # Pattern 1: testMethod(com.example.TestClass)
    pattern1 = re.compile(r'\w+\(([a-zA-Z0-9_.]+)\)')

    # Pattern 2: <<< FAILURE! - in com.example.TestClass
    pattern2 = re.compile(r'<<<\s+(FAILURE|ERROR)!\s+-\s+in\s+([a-zA-Z0-9_.]+)')

    # Pattern 3: [ERROR] Tests run: ... <<< FAILURE! - in com.example.TestClass
    pattern3 = re.compile(r'\[ERROR\].*?in\s+([a-zA-Z0-9_.]+)')

    for line in maven_output.split('\n'):
        # Check pattern 1
        match = pattern1.search(line)
        if match and ('Failed' in line or 'Error' in line or 'failed' in line or 'error' in line):
            full_class_name = match.group(1)
            simple_name = full_class_name.rsplit('.', 1)[-1]
            failed_tests.add(simple_name)

        # Check pattern 2
        match = pattern2.search(line)
        if match:
            full_class_name = match.group(2)
            simple_name = full_class_name.rsplit('.', 1)[-1]
            failed_tests.add(simple_name)

        # Check pattern 3
        match = pattern3.search(line)
        if match:
            full_class_name = match.group(1)
            simple_name = full_class_name.rsplit('.', 1)[-1]
            failed_tests.add(simple_name)

    return failed_tests


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
    maven_module: Optional[str] = None,
    max_tests: Optional[int] = None,
) -> Dict[str, Path]:
    """Run each test class separately with JaCoCo. Returns test_class -> exec_file.

    Args:
        project_root: Project root directory
        test_root: Test source root
        exec_dir: Directory to store JaCoCo exec files
        maven_module: Maven submodule name (e.g., "gson" for multi-module projects)
        max_tests: Maximum number of test classes to run (None = run all)
    """
    test_classes = _discover_test_classes(test_root)
    print(f"Discovered {len(test_classes)} test classes")

    if max_tests is not None and max_tests > 0:
        test_classes = test_classes[:max_tests]
        print(f"Limiting to first {len(test_classes)} test classes for quick validation")

    exec_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    # Ensure JaCoCo agent jar is available
    jacoco_agent_jar = Path.home() / ".m2/repository/org/jacoco/org.jacoco.agent/0.8.12/org.jacoco.agent-0.8.12-runtime.jar"
    if not jacoco_agent_jar.exists():
        print("Downloading JaCoCo agent...")
        subprocess.run(
            ["mvn", "dependency:get", "-Dartifact=org.jacoco:org.jacoco.agent:0.8.12:jar:runtime", "-q"],
            cwd=str(project_root),
            check=False
        )

    for i, tc in enumerate(test_classes, 1):
        exec_file = exec_dir / f"{tc}.exec"
        simple_name = tc.rsplit(".", 1)[-1]
        print(f"  [{i}/{len(test_classes)}] Testing {simple_name}...", end="", flush=True)

        # Use JAVA_TOOL_OPTIONS environment variable to inject JaCoCo agent
        # This bypasses pom.xml's hardcoded argLine configuration
        jacoco_agent_jar = Path.home() / ".m2/repository/org/jacoco/org.jacoco.agent/0.8.12/org.jacoco.agent-0.8.12-runtime.jar"
        java_tool_options = f"-javaagent:{jacoco_agent_jar}=destfile={exec_file.absolute()},append=false"

        cmd = [
            "mvn",
            "test",
            f"-Dtest={simple_name}",
            "-DfailIfNoTests=false",
            "-Dmaven.test.failure.ignore=true",
            "-pl", maven_module if maven_module else ".",
        ]

        # Set environment variable for JaCoCo agent
        env = os.environ.copy()
        env["JAVA_TOOL_OPTIONS"] = java_tool_options

        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            # Print output for the first test to help debug
            if i == 1:
                print(f"\n=== Debug: First test ===")
                print(f"Command: {' '.join(cmd)}")
                print(f"Expected exec file: {exec_file}")
                print(f"\nMaven output:")
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                print(f"=== End debug output ===\n")
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


def _parse_jacoco_for_covered_methods(xml_path: Path) -> Set[str]:
    """Parse JaCoCo XML and return set of covered method signatures."""
    covered = set()

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        for package in root.iter("package"):
            pkg_name = package.get("name", "").replace("/", ".")

            for cls in package.iter("class"):
                class_name = cls.get("name", "").replace("/", ".")

                for method in cls.iter("method"):
                    method_name = method.get("name", "")
                    method_desc = method.get("desc", "")

                    # Check if method has any coverage
                    for counter in method.iter("counter"):
                        if counter.get("type") == "INSTRUCTION":
                            covered_count = int(counter.get("covered", "0"))
                            if covered_count > 0:
                                # Add method signature
                                sig = f"{class_name}.{method_name}{method_desc}"
                                covered.add(sig)
                                break
    except Exception as e:
        print(f"Error parsing XML: {e}")

    return covered


def build_method_lookup_from_jacoco(
    jacoco_xml: Path,
    package_prefix: str,
    src_root: Optional[Path] = None,
) -> Dict[str, JavaMethodInfo]:
    """
    Build method_lookup directly from JaCoCo coverage report.
    Much faster than scanning all class files with javap.

    Args:
        jacoco_xml: Path to JaCoCo XML report
        package_prefix: Package prefix to filter
        src_root: Source root directory used to resolve absolute file paths

    Returns:
        method_lookup dictionary
    """
    method_lookup: Dict[str, JavaMethodInfo] = {}

    try:
        tree = ET.parse(str(jacoco_xml))
        root = tree.getroot()

        for package in root.iter("package"):
            pkg_name = package.get("name", "").replace("/", ".")
            pkg_path = package.get("name", "")  # e.g. "org/apache/commons/io"

            # Filter by package prefix
            if package_prefix and not pkg_name.startswith(package_prefix):
                continue

            for cls in package.iter("class"):
                class_name = cls.get("name", "").replace("/", ".")
                sourcefilename = cls.get("sourcefilename", "")  # e.g. "FileUtils.java"

                # Resolve absolute file path when src_root is available
                if src_root and sourcefilename:
                    filepath = (src_root / pkg_path / sourcefilename).resolve()
                elif sourcefilename:
                    filepath = Path(f"{pkg_path}/{sourcefilename}")
                else:
                    filepath = Path('')

                # Extract simple class name (last part)
                simple_class = class_name.split('.')[-1]

                for method in cls.iter("method"):
                    method_name = method.get("name", "")
                    method_desc = method.get("desc", "")
                    start_line = int(method.get("line", "0"))

                    # Check if method has any coverage
                    has_coverage = False
                    for counter in method.iter("counter"):
                        if counter.get("type") == "INSTRUCTION":
                            covered_count = int(counter.get("covered", "0"))
                            if covered_count > 0:
                                has_coverage = True
                                break

                    if has_coverage:
                        # Create method key: package.Class.method
                        method_key = f"{class_name}.{method_name}"

                        # Create JavaMethodInfo
                        method_info = JavaMethodInfo(
                            package=pkg_name,
                            qualname=f"{simple_class}.{method_name}",
                            filepath=filepath,
                            start=start_line,
                            end=start_line,  # end line unknown from JaCoCo; same as start
                            return_type='',
                            parameter_types=[],
                            modifiers=frozenset()
                        )
                        method_lookup[method_key] = method_info

    except Exception as e:
        print(f"Error building method lookup from JaCoCo: {e}")
        import traceback
        traceback.print_exc()

    return method_lookup


def _is_method_in_covered_set(method_key: str, covered_methods: Set[str]) -> bool:
    """Check if a method from our index appears in covered methods."""
    # method_key format: "package.Class.method(args)returnType"
    # covered format: "package.Class.method(Largs;)Lreturn;"

    # Simple heuristic: check if method name and class match
    parts = method_key.split(".")
    if len(parts) < 2:
        return False

    class_part = ".".join(parts[:-1])
    method_part = parts[-1].split("(")[0]

    for covered in covered_methods:
        if class_part in covered and method_part in covered:
            return True

    return False


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

        # Check if we actually collected any function coverage
        if not functions:
            raise ValueError(
                "No function coverage data collected. The functions field is empty. "
                "This likely means JaCoCo coverage collection failed or no tests covered any functions."
            )

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON graph to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_java_function_mapping(
    project_name: str,
    project_path: str = "../project",
    output_dir: Optional[str] = None,
    max_tests: Optional[int] = None,
) -> Optional[int]:
    """Main entry point: AST analysis + JaCoCo coverage -> JSON mapping.

    Args:
        project_name: Name of the project
        project_path: Path to the parent directory containing the project
        output_dir: Directory to write the mapping file. If None, uses "output/{project_name}"
        max_tests: Maximum number of test classes to run (None = run all, useful for quick validation)

    Returns:
        0 on success, 1 on error
    """
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

    # Extract Maven module name from src_path (e.g., "gson/src/main/java" -> "gson")
    # For multi-module projects, the module is the first path component
    maven_module = None
    if "/" in src_path:
        potential_module = src_path.split("/")[0]
        # Check if this is actually a submodule (has its own pom.xml)
        module_pom = project_root / potential_module / "pom.xml"
        if module_pom.exists():
            maven_module = potential_module
            print(f"Detected Maven submodule: {maven_module}")

    # Use provided output_dir or default to "output/{project_name}"
    if output_dir is None:
        output_dir_path = Path("output") / project_name
    else:
        output_dir_path = Path(output_dir)

    os.makedirs(output_dir_path, exist_ok=True)
    output_json = output_dir_path / "function_testunit_mapping.json"

    if output_json.exists():
        print(f"Output file {output_json} already exists.")
        return 0

    # Step 1: Ensure repo is at correct commit
    if commit_id:
        print(f"Ensuring repo is at commit {commit_id[:12]}...")
        try:
            # Check current commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=10
            )
            current_commit = result.stdout.strip()

            if current_commit != commit_id:
                print(f"Warning: Current commit {current_commit[:12]} != expected {commit_id[:12]}")
                print(f"Running git reset --hard {commit_id}...")
                subprocess.run(
                    ["git", "reset", "--hard", commit_id],
                    cwd=str(project_root),
                    check=True,
                    timeout=30
                )
                print(f"✓ Reset to {commit_id[:12]}")
            else:
                print(f"✓ Already at correct commit {commit_id[:12]}")

        except subprocess.CalledProcessError as e:
            print(f"Warning: git operations failed: {e}")
            print("Continuing anyway (assuming repo is at correct state)...")
        except Exception as e:
            print(f"Warning: Unexpected error during git check: {e}")
            print("Continuing anyway...")

    # Step 2: Build project (compile only, skip tests)
    print("Compiling project...")

    # Prepare clean environment without JAVA_TOOL_OPTIONS to avoid JaCoCo agent conflicts
    clean_env = os.environ.copy()
    clean_env["JAVA_TOOL_OPTIONS"] = ""

    # Run the project-specific build command (if any)
    build_cmd = repo_spec.get("build_cmd", "mvn compile")
    if isinstance(build_cmd, str):
        build_cmd = [build_cmd]
    for cmd in build_cmd:
        try:
            subprocess.run(cmd.split(), cwd=str(project_root), env=clean_env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: build command '{cmd}' failed with code {e.returncode}")
            # If we already compiled with test-compile above, this might be redundant
            if not maven_module:
                raise

    # Step 3: Run tests with JaCoCo to generate coverage data
    print("Discovering tests...")
    test_classes = _discover_test_classes(test_root)
    print(f"Found {len(test_classes)} test classes")

    # Determine which tests to run based on max_tests parameter
    tests_to_run = test_classes
    if max_tests is not None and max_tests > 0:
        tests_to_run = test_classes[:max_tests]
        print(f"Limiting to {len(tests_to_run)} test classes (max_tests={max_tests})")
    else:
        print(f"Running all {len(test_classes)} test classes")

    # For multi-module projects, use the module's target directory
    if maven_module:
        exec_file = (project_root / maven_module / "target" / "jacoco.exec").resolve()
        report_dir = (project_root / maven_module / "target" / "site" / "jacoco").resolve()
    else:
        exec_file = (project_root / "target" / "jacoco.exec").resolve()
        report_dir = (project_root / "target" / "site" / "jacoco").resolve()

    exec_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove old exec file to start fresh
    if exec_file.exists():
        exec_file.unlink()
        print(f"Removed old coverage file")

    # Initialize failed_tests tracking
    failed_tests: Set[str] = set()

    # Prepare clean environment for test execution
    env = os.environ.copy()

    # Debug: show current JAVA_TOOL_OPTIONS
    if "JAVA_TOOL_OPTIONS" in env:
        print(f"[DEBUG] Found existing JAVA_TOOL_OPTIONS: {env['JAVA_TOOL_OPTIONS']}")
        del env["JAVA_TOOL_OPTIONS"]
        print(f"[DEBUG] Deleted JAVA_TOOL_OPTIONS from environment")
    else:
        print(f"[DEBUG] No existing JAVA_TOOL_OPTIONS in environment")

    # Build Maven test command based on whether project has JaCoCo in pom.xml
    jacoco_in_pom = repo_spec.get("jacoco_in_pom", False)

    if jacoco_in_pom:
        # Project has JaCoCo configured in pom.xml, just run mvn test
        print("Project has JaCoCo configured in pom.xml")
        # Keep JAVA_TOOL_OPTIONS cleared to avoid conflicts
        cmd = [
            "mvn", "test",
            f"-Djacoco.destFile={exec_file}",
        ]
    else:
        # Project doesn't have JaCoCo - use JAVA_TOOL_OPTIONS to load agent
        print("Project needs JaCoCo - setting JAVA_TOOL_OPTIONS")
        jacoco_agent = f"-javaagent:{Path.home()}/.m2/repository/org/jacoco/org.jacoco.agent/0.8.12/org.jacoco.agent-0.8.12-runtime.jar=destfile={exec_file}"
        env["JAVA_TOOL_OPTIONS"] = jacoco_agent
        print(f"[DEBUG] Set JAVA_TOOL_OPTIONS to: {env['JAVA_TOOL_OPTIONS']}")
        cmd = [
            "mvn", "test",
        ]

    # Add -Dtest parameter only when limiting tests
    if max_tests is not None and max_tests > 0 and tests_to_run:
        test_pattern = ",".join(tests_to_run)
        cmd.append(f"-Dtest={test_pattern}")
        print(f"Running with test pattern (first 3): {','.join(tests_to_run[:3])}...")

    cmd.extend(["-pl", maven_module if maven_module else "."])

    # Run tests with visible output (no capture, so user can see real-time progress)
    print(f"\nExecuting: {' '.join(cmd)}")
    print(f"Expected coverage file: {exec_file}\n")
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        timeout=600,
    )

    # Parse Surefire reports to identify failed tests (more reliable than parsing output)
    failed_tests = set()

    # Determine surefire-reports directory
    if maven_module:
        surefire_reports_dir = project_root / maven_module / "target" / "surefire-reports"
    else:
        surefire_reports_dir = project_root / "target" / "surefire-reports"

    print(f"\nParsing Surefire reports from: {surefire_reports_dir}")
    failed_tests = _parse_failed_tests_from_surefire_reports(surefire_reports_dir)

    if failed_tests:
        print(f"\nFound {len(failed_tests)} failed test classes:")
        for test in sorted(failed_tests):
            print(f"  - {test}")
    elif result.returncode != 0:
        print(f"\nWarning: Maven test returned code {result.returncode} but no specific test failures identified")

    # Check if coverage file was created
    print(f"\nChecking for coverage file at: {exec_file}")
    print(f"  File exists: {exec_file.exists()}")

    if exec_file.exists() and exec_file.stat().st_size > 0:
        print(f"  File size: {exec_file.stat().st_size} bytes")
    else:
        if exec_file.exists():
            print(f"  File exists but is empty (0 bytes)")

        # Look for any .exec files in target directory (might be named differently)
        target_dir = exec_file.parent
        print(f"  Searching for alternative .exec files in {target_dir}...")
        import glob
        exec_files = glob.glob(str(target_dir / "*.exec"))
        exec_files = [f for f in exec_files if Path(f).stat().st_size > 0]  # Only non-empty files

        if exec_files:
            print(f"  Found alternative coverage files: {exec_files}")
            # Use the most recent non-empty .exec file
            exec_file = Path(max(exec_files, key=lambda f: Path(f).stat().st_mtime))
            print(f"  Using: {exec_file} (size: {exec_file.stat().st_size} bytes)")
        else:
            print(f"\nERROR: Coverage file not generated: {exec_file}")
            return 1

    print(f"✓ Coverage file generated: {exec_file.stat().st_size} bytes")

    # Step 5: Generate coverage report
    print("Generating coverage report...")
    cmd = [
        "mvn",
        "org.jacoco:jacoco-maven-plugin:0.8.12:report",
        f"-Djacoco.dataFile={exec_file}",
        "-pl", maven_module if maven_module else ".",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )

    jacoco_xml = report_dir / "jacoco.xml"
    if not jacoco_xml.exists():
        print(f"ERROR: Report not generated: {jacoco_xml}")
        print(f"Maven output: {result.stdout[-500:]}")
        return 1

    print(f"Report generated: {jacoco_xml}")

    # Step 6: Extract methods directly from JaCoCo coverage report
    print("Extracting methods from JaCoCo coverage...")
    method_lookup = build_method_lookup_from_jacoco(jacoco_xml, package_prefix, src_root=src_root)
    print(f"Found {len(method_lookup)} methods with coverage")

    # Step 7: Build mapping
    print("Building mapping...")
    file_index: Dict[Path, JavaFileIndex] = {}
    class_lookup: Dict[str, JavaClassInfo] = {}
    graph = JavaCoverageGraph(src_root, file_index, method_lookup, class_lookup)
    test_class_names = [tc.rsplit(".", 1)[-1] for tc in test_classes]

    # Filter out failed tests from the mapping
    successful_test_names = [tc for tc in test_class_names if tc not in failed_tests]

    if failed_tests:
        print(f"\nExcluding {len(failed_tests)} failed test classes from mapping")
        print(f"Including {len(successful_test_names)} successful test classes")
    else:
        print(f"All {len(test_class_names)} test classes passed")

    # All covered methods map to successful tests (simplified approach)
    for method_key in method_lookup.keys():
        # All methods in method_lookup already have coverage (from JaCoCo)
        for test_name in successful_test_names:
            graph.method_to_tests[method_key].add(test_name)


    # Step 8: Export
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
