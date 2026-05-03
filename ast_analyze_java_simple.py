"""
Simplified Java function mapping - runs all tests once and generates coverage.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set
import xml.etree.ElementTree as ET

from ast_analyze_java import (
    build_java_function_index,
    get_spec,
    _discover_test_classes,
)


def generate_simple_mapping(
    project_name: str,
    project_path: str = "../project",
    output_dir: str = None,
    max_tests: int = None,
) -> int:
    """Generate function mapping by running all tests once.

    Args:
        project_name: Project name
        project_path: Path to project directory
        output_dir: Output directory for JSON
        max_tests: Maximum number of tests to run (None = all)
    """

    repo_spec = get_spec(project_name)
    if not repo_spec:
        print(f"Project {project_name} not found")
        return 1

    project_root = Path(project_path) / project_name
    src_path = repo_spec.get("src_path", "src/main/java")
    test_path = repo_spec.get("test_path", "src/test/java")
    package_prefix = repo_spec.get("package_prefix", "")
    commit_id = repo_spec.get("commit_id", "")

    src_root = (project_root / src_path).resolve()
    test_root = (project_root / test_path).resolve()

    # Detect Maven module
    maven_module = None
    if "/" in src_path:
        potential_module = src_path.split("/")[0]
        if (project_root / potential_module / "pom.xml").exists():
            maven_module = potential_module

    print(f"Project: {project_name}")
    print(f"Maven module: {maven_module}")

    # Step 1: Build function index
    print("Building function index...")
    file_index, method_lookup, class_lookup = build_java_function_index(
        src_root, package_prefix
    )
    print(f"Found {len(method_lookup)} methods")

    # Step 2: Discover test classes
    print("Discovering tests...")
    test_classes = _discover_test_classes(test_root)
    print(f"Found {len(test_classes)} test classes")

    if max_tests is not None and max_tests > 0:
        test_classes = test_classes[:max_tests]
        print(f"Limiting to first {max_tests} tests")

    # Step 3: Run tests with JaCoCo
    print(f"Running {len(test_classes)} tests with coverage...")

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
        print(f"Removed old coverage file: {exec_file}")

    # Set JAVA_TOOL_OPTIONS for JaCoCo
    jacoco_agent = Path.home() / ".m2/repository/org/jacoco/org.jacoco.agent/0.8.12/org.jacoco.agent-0.8.12-runtime.jar"
    env = os.environ.copy()
    # Use append=true to accumulate coverage from all test classes
    env["JAVA_TOOL_OPTIONS"] = f"-javaagent:{jacoco_agent}=destfile={exec_file},append=true"

    # According to gson's pom.xml: <skip>${maven.test.skip}</skip>
    # But we want to run tests, just skip ProGuard obfuscation
    # Solution: Use Maven profile or skip specific execution ID

    # Build test pattern for specific tests
    if test_classes:
        test_pattern = ",".join([tc.rsplit(".", 1)[-1] for tc in test_classes])
        cmd = [
            "mvn", "test",
            f"-Dtest={test_pattern}",
            "-DfailIfNoTests=false",
            "-pl", maven_module if maven_module else "."
        ]
    else:
        cmd = [
            "mvn", "test",
            "-pl", maven_module if maven_module else "."
        ]

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if not exec_file.exists():
        print(f"ERROR: Coverage file not generated: {exec_file}")
        return 1

    print(f"Coverage file generated: {exec_file.stat().st_size} bytes")

    # Step 4: Generate report
    print("Generating coverage report...")
    print(f"  Using exec_file: {exec_file}")
    print(f"  Report will go to: {report_dir}")

    cmd = [
        "mvn",
        "org.jacoco:jacoco-maven-plugin:0.8.12:report",
        f"-Djacoco.dataFile={exec_file}",
        "-pl", maven_module if maven_module else ".",
    ]
    print(f"  Command: {' '.join(cmd)}")

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

    # Step 5: Parse coverage - extract covered classes/methods
    print("Parsing coverage...")
    covered_methods = parse_jacoco_coverage(jacoco_xml)
    print(f"Found {len(covered_methods)} covered methods")

    # Step 6: Build mapping - all covered methods -> all tests
    print("Building mapping...")

    # Create mapping: each covered method maps to all test classes
    # Format compatible with Python version
    functions = {}
    test_class_names = [tc.rsplit(".", 1)[-1] for tc in test_classes]

    for method_key, method_info in method_lookup.items():
        # Check if this method was covered
        if is_method_covered(method_key, covered_methods):
            # Map to all test classes (simple approximation)
            functions[method_key] = {
                "file": str(method_info.filepath),
                "relative_file": str(method_info.filepath.relative_to(src_root)),
                "line_range": [method_info.start, method_info.end],
                "tests": test_class_names,
                "coverage_type": "aggregate"  # Mark as simplified mapping
            }

    print(f"Mapped {len(functions)} functions to tests")

    # Check if we collected any function coverage
    if not functions:
        print("ERROR: No function coverage collected. Functions field is empty.")
        print("This likely means:")
        print("  1. JaCoCo coverage collection failed")
        print("  2. No methods were covered by tests")
        print("  3. ProGuard obfuscation prevented coverage collection")
        return 1

    # Step 7: Export JSON
    if output_dir is None:
        output_dir = f"output/{project_name}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_json = Path(output_dir) / "function_testunit_mapping.json"

    payload = {
        "meta": {
            "src_path": src_path,
            "test_path": test_path,
            "commit_id": commit_id,
            "language": "java",
            "note": "Simplified mapping - all covered methods map to all tests",
        },
        "functions": functions,
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote mapping to {output_json}")
    print(f"SUCCESS: {len(functions)} functions mapped")
    return 0


def parse_jacoco_coverage(xml_path: Path) -> Set[str]:
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


def is_method_covered(method_key: str, covered_methods: Set[str]) -> bool:
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


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "gson"
    max_tests = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Running with max_tests={max_tests}")
    exit(generate_simple_mapping(project, "../project", f"output/{project}", max_tests=max_tests))
