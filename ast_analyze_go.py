"""
Go function-to-test mapping using coverage analysis.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class GoFunctionInfo:
    """Information about a Go function."""
    name: str
    filepath: Path
    start_line: int
    end_line: int
    receiver: str = None  # For methods, e.g., "*Context"


def get_spec(project_name: str) -> dict:
    """Load project spec from repo_list.json."""
    repo_list_path = Path(__file__).parent / "repo_list.json"
    with open(repo_list_path) as f:
        repos = json.load(f)
    return repos.get(project_name)


def build_go_function_index(src_root: Path, package_prefix: str) -> Dict[str, GoFunctionInfo]:
    """Build index of all functions in Go source files.

    Returns:
        Dict mapping "package.FunctionName" or "package.Type.MethodName" to function info
    """
    function_index = {}

    # Find all .go files (excluding _test.go)
    for go_file in src_root.rglob("*.go"):
        if go_file.name.endswith("_test.go"):
            continue

        try:
            with open(go_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue

        # Extract package name
        pkg_match = re.search(r'^package\s+(\w+)', content, re.MULTILINE)
        if not pkg_match:
            continue
        pkg_name = pkg_match.group(1)

        # Find all function declarations
        # Matches: func Name(...) or func (receiver Type) Name(...)
        func_pattern = r'^func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\('

        lines = content.split('\n')
        for i, line in enumerate(lines, start=1):
            match = re.match(func_pattern, line)
            if match:
                receiver_name, receiver_type, func_name = match.groups()

                # Skip private functions (lowercase first letter) unless they're test helpers
                if func_name[0].islower() and not func_name.startswith("test"):
                    continue

                # Find end line (simplified - just look for closing brace)
                end_line = find_function_end(lines, i - 1)

                # Build function key
                if receiver_type:
                    # Method: "package.Type.MethodName"
                    func_key = f"{pkg_name}.{receiver_type}.{func_name}"
                else:
                    # Function: "package.FunctionName"
                    func_key = f"{pkg_name}.{func_name}"

                function_index[func_key] = GoFunctionInfo(
                    name=func_name,
                    filepath=go_file,
                    start_line=i,
                    end_line=end_line,
                    receiver=receiver_type
                )

    return function_index


def find_function_end(lines: List[str], start_idx: int) -> int:
    """Find the end line of a function by counting braces."""
    depth = 0
    found_open = False

    for i in range(start_idx, len(lines)):
        line = lines[i]
        # Simple brace counting (doesn't handle strings/comments perfectly)
        for ch in line:
            if ch == '{':
                depth += 1
                found_open = True
            elif ch == '}':
                depth -= 1
                if found_open and depth == 0:
                    return i + 1  # 1-indexed

    return start_idx + 1


def discover_test_functions(test_root: Path) -> List[str]:
    """Discover all test functions in *_test.go files.

    Returns:
        List of test function names (e.g., ["TestGinRun", "TestContextGet"])
    """
    test_functions = []

    for test_file in test_root.rglob("*_test.go"):
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue

        # Find test functions: func TestXxx(t *testing.T)
        test_pattern = r'^func\s+(Test\w+)\s*\('
        for match in re.finditer(test_pattern, content, re.MULTILINE):
            test_name = match.group(1)
            test_functions.append(test_name)

    return test_functions


def run_tests_with_coverage(project_root: Path, coverage_file: Path) -> bool:
    """Run all tests with coverage collection.

    Args:
        project_root: Root directory of Go project
        coverage_file: Output path for coverage profile

    Returns:
        True if tests ran successfully
    """
    # Remove old coverage file
    if coverage_file.exists():
        coverage_file.unlink()

    coverage_file.parent.mkdir(parents=True, exist_ok=True)

    # Run tests with coverage for all packages
    cmd = [
        "go", "test",
        "-coverprofile=" + str(coverage_file),
        "-covermode=set",
        "./...",  # All packages
    ]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"Warning: tests failed with exit code {result.returncode}")
        print(f"Stderr: {result.stderr[-500:]}")

    return coverage_file.exists()


def parse_go_coverage(coverage_file: Path, src_root: Path) -> Dict[Path, Set[int]]:
    """Parse Go coverage profile to get covered lines per file.

    Returns:
        Dict mapping file paths to set of covered line numbers
    """
    covered_lines = {}

    try:
        with open(coverage_file, 'r') as f:
            for line in f:
                # Skip mode line
                if line.startswith("mode:"):
                    continue

                # Format: github.com/gin-gonic/gin/binding.go:45.13,47.2 1 1
                # Last number is count (0 = not covered, >0 = covered)
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                location = parts[0]
                count = int(parts[2])

                if count == 0:
                    continue  # Not covered

                # Parse file:startLine.startCol,endLine.endCol
                match = re.match(r'(.+):(\d+)\.\d+,(\d+)\.\d+', location)
                if not match:
                    continue

                file_path, start_line, end_line = match.groups()
                start_line = int(start_line)
                end_line = int(end_line)

                # Convert package path to file path
                # e.g., github.com/gin-gonic/gin/binding.go -> ./binding.go
                file_name = file_path.split('/')[-1]

                # Find the actual file in src_root
                for actual_file in src_root.rglob(f"**/{file_name}"):
                    if actual_file not in covered_lines:
                        covered_lines[actual_file] = set()

                    # Add all lines in range
                    for line_num in range(start_line, end_line + 1):
                        covered_lines[actual_file].add(line_num)
                    break

    except Exception as e:
        print(f"Error parsing coverage: {e}")

    return covered_lines


def is_function_covered(func_info: GoFunctionInfo, covered_lines: Dict[Path, Set[int]]) -> bool:
    """Check if a function has any coverage."""
    if func_info.filepath not in covered_lines:
        return False

    file_covered = covered_lines[func_info.filepath]

    # Check if any line in the function's range is covered
    for line_num in range(func_info.start_line, func_info.end_line + 1):
        if line_num in file_covered:
            return True

    return False


def generate_go_function_mapping(
    project_name: str,
    project_path: str = "../project",
    output_dir: str = None,
    max_tests: int = None,
) -> int:
    """Generate function-to-test mapping for Go project.

    Args:
        project_name: Project name
        project_path: Path to project directory
        output_dir: Output directory for JSON
        max_tests: Not used for Go (runs all tests together)
    """
    repo_spec = get_spec(project_name)
    if not repo_spec:
        print(f"Project {project_name} not found")
        return 1

    if repo_spec.get("language") != "go":
        print(f"Project {project_name} is not a Go project")
        return 1

    project_root = Path(project_path) / project_name
    src_path = repo_spec.get("src_path", ".")
    commit_id = repo_spec.get("commit_id", "")
    package_prefix = repo_spec.get("package_prefix", "")

    src_root = (project_root / src_path).resolve()

    print(f"Project: {project_name}")
    print(f"Source root: {src_root}")

    # Step 1: Build function index
    print("Building function index...")
    function_index = build_go_function_index(src_root, package_prefix)
    print(f"Found {len(function_index)} functions")

    # Step 2: Discover test functions
    print("Discovering test functions...")
    test_functions = discover_test_functions(src_root)
    print(f"Found {len(test_functions)} test functions")

    if not test_functions:
        print("Warning: No test functions found")

    # Step 3: Run tests with coverage
    print("Running tests with coverage...")
    coverage_file = project_root / "coverage.out"

    if not run_tests_with_coverage(project_root, coverage_file):
        print("ERROR: Failed to generate coverage file")
        return 1

    print(f"Coverage file generated: {coverage_file}")

    # Step 4: Parse coverage
    print("Parsing coverage...")
    covered_lines = parse_go_coverage(coverage_file, src_root)
    print(f"Found coverage for {len(covered_lines)} files")

    # Step 5: Build mapping
    print("Building function-to-test mapping...")
    functions = {}

    for func_key, func_info in function_index.items():
        if is_function_covered(func_info, covered_lines):
            # Map to all test functions (simplified approach)
            functions[func_key] = {
                "file": str(func_info.filepath),
                "relative_file": str(func_info.filepath.relative_to(src_root)),
                "line_range": [func_info.start_line, func_info.end_line],
                "tests": test_functions,
                "coverage_type": "aggregate"
            }

    print(f"Mapped {len(functions)} functions to tests")

    # Check if we have any mappings
    if not functions:
        print("ERROR: No function coverage collected. Functions field is empty.")
        return 1

    # Step 6: Export JSON
    if output_dir is None:
        output_dir = f"output/{project_name}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_json = Path(output_dir) / "function_testunit_mapping.json"

    payload = {
        "meta": {
            "src_path": src_path,
            "test_path": src_path,  # Go tests are in same directory
            "commit_id": commit_id,
            "language": "go",
            "note": "Simplified mapping - all covered functions map to all tests",
        },
        "functions": functions,
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote mapping to {output_json}")
    print(f"SUCCESS: {len(functions)} functions mapped")
    return 0


if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else "gin"
    exit(generate_go_function_mapping(project, "../project", f"output/{project}"))
